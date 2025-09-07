from __future__ import annotations
import os, sys, json, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict


import numpy as np
import pandas as pd
from tqdm import tqdm


# Geo
import geopandas as gpd
from shapely.geometry import box
import rasterio
import rioxarray as rxr
import xarray as xr


# ML
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from catboost import CatBoostRanker

# Hyperbolic taxonomy embeddings
from gensim.models.poincare import PoincareModel


# R interface for FishBase
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()


R = ro.r

@dataclass
class Config:
    country_code: str = "VNM" # ISO3
    ssp: str = "245" # "245" or "585"
    year_future: int = 2050 # 2050 or 2100
    systems: Tuple[str,...] = ("pond","cage")
    data_dir: Path = Path("data")
    out_dir: Path = Path("outputs")
    
# ---------- Utilities ----------


def ensure_dirs(cfg: Config):
    for p in [cfg.data_dir/"fishbase", cfg.data_dir/"env", cfg.data_dir/"admin", cfg.data_dir/"faostat",
            cfg.out_dir/"models", cfg.out_dir/"tables", cfg.out_dir/"figures"]:
        p.mkdir(parents=True, exist_ok=True)

def fishbase_pull(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Pull core FishBase tables with rfishbase and convert to pandas."""
    try:
        rfb = importr("rfishbase")
    except Exception as e:
        raise RuntimeError("rfishbase is required in R environment.") from e


    # Tables
    species = ro.r('rfishbase::species()')
    ecology = ro.r('rfishbase::ecology()')
    popgrowth = ro.r('rfishbase::popgrowth()')
    country = ro.r('rfishbase::country()')


    # Optional/case-sensitive tables often available in fb_tbl
    fb_tbl = ro.r('rfishbase::fb_tbl')
    try:
        countrysub = fb_tbl("countrysub")
    except Exception:
        countrysub = ro.r('data.frame()')
    try:
        cultsys = fb_tbl("cultsys")
    except Exception:
        cultsys = ro.r('data.frame()')


    # Convert to pandas
    out = {
        'species': pandas2ri.rpy2py(species),
        'ecology': pandas2ri.rpy2py(ecology),
        'popgrowth': pandas2ri.rpy2py(popgrowth),
        'country': pandas2ri.rpy2py(country),
        'countrysub': pandas2ri.rpy2py(countrysub),
        'cultsys': pandas2ri.rpy2py(cultsys),
    }
    # Basic selects/renames for downstream
    for k,df in out.items():
        if 'SpecCode' in df.columns:
            df['SpecCode'] = df['SpecCode'].astype('Int64')
    return out

def load_vietnam_admin(cfg: Config) -> gpd.GeoDataFrame:
    """Load Viet Nam provinces from a local GeoPackage/Shapefile.
    Expect a file at data/admin/vietnam_provinces.gpkg with a 'NAME_1' (province) column."""
    # You can replace this with any L1/L2 admin file you have.
    candidates = [cfg.data_dir/"admin"/"vietnam_provinces.gpkg",
                cfg.data_dir/"admin"/"vietnam_provinces.shp"]
    for c in candidates:
        if c.exists():
            gdf = gpd.read_file(c)
            return gdf.to_crs(4326)
    raise FileNotFoundError("Place a provinces file at data/admin/vietnam_provinces.gpkg or .shp")

def load_env_layers(cfg: Config) -> Dict[str, xr.DataArray]:
    """Load environmental rasters as xarray DataArrays with rioxarray.
    Expected files in data/env/ e.g., BO_sstmean_ss.tif, BO_salinity_ss.tif, and future variants.
    """
    layers = {}
    for key, fname in {
        'sst_now': 'BO_sstmean_ss.tif',
        'sss_now': 'BO_salinity_ss.tif',
        # Future examples (adjust to your filenames)
        'sst_2050_245': 'BO_sstmean_ssp245_2041-2060_ss.tif',
        'sss_2050_245': 'BO_salinity_ssp245_2041-2060_ss.tif',
        'sst_2050_585': 'BO_sstmean_ssp585_2041-2060_ss.tif',
        'sss_2050_585': 'BO_salinity_ssp585_2041-2060_ss.tif',
    }.items():
        fpath = cfg.data_dir/"env"/fname
        if fpath.exists():
            arr = rxr.open_rasterio(fpath, masked=True).squeeze(drop=True)
            layers[key] = arr
    if not layers:
        raise FileNotFoundError("No Bio-ORACLE rasters found in data/env. See README in this file.")
    return layers

def province_stats(gdf: gpd.GeoDataFrame, arr: xr.DataArray, stat: str="mean") -> pd.Series:
    """Compute province-level statistic for a raster (simple masked mean)."""
    values = []
    for geom in gdf.geometry:
        clipped = arr.rio.clip([geom.__geo_interface__], gdf.crs, drop=True)
        dat = clipped.values
        dat = dat[np.isfinite(dat)]
        if dat.size == 0:
            values.append(np.nan)
        else:
            if stat == 'mean':
                values.append(float(np.nanmean(dat)))
            elif stat == 'p10':
                values.append(float(np.nanpercentile(dat, 10)))
            elif stat == 'p90':
                values.append(float(np.nanpercentile(dat, 90)))
            else:
                values.append(float(np.nanmean(dat)))
    return pd.Series(values, index=gdf.index)

def build_feature_table(fb: Dict[str,pd.DataFrame], provinces: gpd.GeoDataFrame, env: Dict[str,xr.DataArray], cfg: Config) -> pd.DataFrame:
    # Traits
    sp = fb['species'][['SpecCode','Genus','Species','FBname','AnaCat','DemersPelag']].copy()
    ecol = fb['ecology'].copy()
    cols_keep = [c for c in ['SpecCode','Tempmin','Tempmax','DepthRangeDeep','DepthRangeShallow','Saltwater'] if c in ecol.columns]
    ecol = ecol[cols_keep].drop_duplicates('SpecCode')
    pop = fb['popgrowth'][['SpecCode','Linf','K','to']].drop_duplicates('SpecCode') if 'popgrowth' in fb else pd.DataFrame()


    traits = sp.merge(ecol, on='SpecCode', how='left').merge(pop, on='SpecCode', how='left')


    # Province env stats (present)
    provinces = provinces.copy()
    if 'sst_now' in env: provinces['sst_mean'] = province_stats(provinces, env['sst_now'], 'mean')
    if 'sss_now' in env: provinces['sss_mean'] = province_stats(provinces, env['sss_now'], 'mean')


    # Long-form cartesian product: species × province × system
    species_df = traits.dropna(subset=['SpecCode']).copy()
    species_df['species_key'] = species_df['Genus'].str.cat(species_df['Species'], sep=' ')
    prov_df = provinces[['NAME_1']].reset_index(drop=True).rename(columns={'NAME_1':'province'})


    systems = list(cfg.systems)
    idx = pd.MultiIndex.from_product([species_df['SpecCode'].values, prov_df['province'].values, systems], names=['SpecCode','province','system'])
    X = pd.DataFrame(index=idx).reset_index()


    X = X.merge(species_df, on='SpecCode', how='left')
    X = X.merge(prov_df, on='province', how='left')
    X = X.merge(provinces[['NAME_1','sst_mean','sss_mean']].rename(columns={'NAME_1':'province'}), on='province', how='left')


    # Simple climate-overlap heuristics as features
    if {'Tempmin','Tempmax'}.issubset(X.columns):
        X['temp_mid'] = (X['Tempmin'] + X['Tempmax'])/2
        X['temp_gap'] = (X['sst_mean'] - X['temp_mid']).abs()
    if 'sss_mean' in X.columns and 'Saltwater' in X.columns:
    # Encode rough salinity compatibility: 1 if marine/brackish near saline provinces, else 0 (very rough)
        X['salinity_flag'] = np.where((X['Saltwater'] == 'marine') & (X['sss_mean'] > 30), 1,
                                np.where((X['Saltwater'] == 'freshwater') & (X['sss_mean'] < 5), 1,
                                np.where((X['Saltwater'] == 'brackish') & (X['sss_mean'].between(5,30)), 1, 0)))


    # System interactions
    X['is_pond'] = (X['system'] == 'pond').astype(int)
    X['is_cage'] = (X['system'] == 'cage').astype(int)


    return X

def build_pu_labels(fb: Dict[str,pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Create country-level positives from FAO FishStat or CULTSYS (weak labels), then broadcast to provinces by presence.
    Expect a CSV at data/faostat/production_aquaculture.csv with columns [country_iso3, species, year, quantity]."""
    fao_path = cfg.data_dir/"faostat"/"production_aquaculture.csv"
    if not fao_path.exists():
        warnings.warn("FAO aquaculture CSV missing; labels will rely only on FishBase CULTSYS if available.")
        fao = pd.DataFrame(columns=['country_iso3','species','year','quantity'])
    else:
        fao = pd.read_csv(fao_path)


    cult = fb.get('cultsys', pd.DataFrame()).copy()
    # Normalize species names for join
    species = fb['species'][['SpecCode','Genus','Species']].copy()
    species['species_name'] = species['Genus'].str.cat(species['Species'], sep=' ')


    pos_species_country = set()
    # FAO positives (country level)
    if not fao.empty:
        fao_vn = fao.loc[fao['country_iso3'] == cfg.country_code]
        # threshold for positive (ever produced >= some tonnage)
        thresh = np.nanpercentile(fao_vn['quantity'].fillna(0), 50) if not fao_vn.empty else 0
        fao_pos = set(fao_vn.loc[fao_vn['quantity'] >= thresh, 'species'].str.strip().str.lower().unique())
        pos_species_country |= fao_pos


    # CULTSYS positives (global → assume available in country if species occurs there)
    if not cult.empty and 'Speccode' in cult.columns:
    # Standardize key capitalization
        cult = cult.rename(columns={'Speccode':'SpecCode'})
        cult = cult.dropna(subset=['SpecCode']).drop_duplicates('SpecCode')
        cult_sp = species.merge(cult[['SpecCode']], on='SpecCode', how='inner')
        pos_species_country |= set(cult_sp['species_name'].str.lower().unique())


    labels = pd.DataFrame({'species_name': list(pos_species_country)})
    labels['label'] = 1
    return labels

def prepare_ltr_matrix(X: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X = X.copy()
    # Merge weak positives; unlabeled = 0
    X['species_name'] = X['Genus'].str.cat(X['Species'], sep=' ').str.lower()
    X = X.merge(labels, on='species_name', how='left')
    X['y_pu'] = X['label'].fillna(0).astype(int)


    # Group id = province + system for LTR
    X['group_id'] = X['province'].astype(str) + '|' + X['system'].astype(str)


    # Minimal feature set (extend as needed)
    num_feats = [c for c in ['temp_gap','salinity_flag','Linf','K','to','sst_mean','sss_mean'] if c in X.columns]
    cat_feats = [c for c in ['AnaCat','DemersPelag','system'] if c in X.columns]


    # One-hot encode categories
    ct = ColumnTransformer([
        ("num", 'passthrough', num_feats),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])


    X_numcat = ct.fit_transform(X)
    feature_names = list(num_feats) + list(ct.named_transformers_['cat'].get_feature_names_out(cat_feats))


    # Targets for LTR: use PU labels as relevance (weak). In practice, replace with PU-probabilities.
    y = X['y_pu'].values


    # group sizes
    groups = X.groupby('group_id').size().values


    return X, X_numcat, y, groups

def train_lightgbm_ranker(X_mat, y, groups):
    params = dict(
        objective='lambdarank',
        metric='ndcg',
        boosting_type='gbdt',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model = lgb.LGBMRanker(**params)
    model.fit(X_mat, y, group=groups)
    return model

def learn_taxonomy_embeddings(fb: Dict[str,pd.DataFrame], size: int=20) -> Dict[str,np.ndarray]:
    sp = fb['species'][['SpecCode','Genus','Species']].copy()
    sp['species_name'] = sp['Genus'].str.cat(sp['Species'], sep=' ')
    # Build edges for a simple tree: Family->Genus, Genus->Species (requires family)
    taxa_cols = [c for c in ['Class','Order','Family','Genus','Species'] if c in fb['species'].columns]
    if 'Family' not in taxa_cols:
        return {}
    edges = []
    df = fb['species'][['Family','Genus','Species']].dropna()
    edges += list({(f"Family:{r.Family}", f"Genus:{r.Genus}") for r in df.itertuples(index=False)})
    edges += list({(f"Genus:{r.Genus}", f"Species:{r.Species}") for r in df.itertuples(index=False)})


    model = PoincareModel(edges, size=size, negative=10)
    model.train(epochs=50)


    # Extract species vectors
    emb = {}
    for r in df.itertuples(index=False):
        key = f"Species:{r.Species}"
        if key in model.kv:
            emb[r.Species] = model.kv[key]
    return emb

def ndcg_at_k(y_true, y_score, groups, k=10):
    # Compute NDCG@k per group and average
    res = []
    start = 0
    for g in groups:
        y_t = y_true[start:start+g]
        y_s = y_score[start:start+g]
        order = np.argsort(-y_s)
        gains = (2**y_t[order] - 1)
        discounts = 1/np.log2(np.arange(2, 2+min(k,g)))
        dcg = np.sum(gains[:k] * discounts[:len(gains[:k])])
        # ideal
        order_i = np.argsort(-y_t)
        gains_i = (2**y_t[order_i] - 1)
        idcg = np.sum(gains_i[:k] * discounts[:len(gains_i[:k])]) if gains_i[:k].sum()>0 else 1.0
        res.append(dcg/idcg)
        start += g
    return float(np.mean(res))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--country_code', default='VNM')
    ap.add_argument('--ssp', default='245')
    ap.add_argument('--year_future', type=int, default=2050)
    ap.add_argument('--systems', nargs='+', default=['pond','cage'])
    cfg = Config(country_code=ap.parse_args().country_code,
        ssp=ap.parse_args().ssp,
        year_future=ap.parse_args().year_future,
        systems=tuple(ap.parse_args().systems))

    ensure_dirs(cfg)


    print("[1/6] Pulling FishBase tables via rpy2…")
    fb = fishbase_pull(cfg)


    print("[2/6] Loading admin provinces…")
    provinces = load_vietnam_admin(cfg)


    print("[3/6] Loading environmental rasters…")
    env = load_env_layers(cfg)


    print("[4/6] Building features (species × province × system)…")
    X = build_feature_table(fb, provinces, env, cfg)


    print("[5/6] Building weak PU labels…")
    labels = build_pu_labels(fb, cfg)


    print("[6/6] Preparing LTR matrix & training…")
    X_full, X_mat, y, groups = prepare_ltr_matrix(X, labels)
    model = train_lightgbm_ranker(X_mat, y, groups)
    scores = model.predict(X_mat)


    ndcg10 = ndcg_at_k(y, scores, groups, k=10)
    print(f"NDCG@10 (weak labels): {ndcg10:.3f}")


    # Export top-k per province/system
    X_full['score'] = scores
    topk = (X_full
    .sort_values(['province','system','score'], ascending=[True,True,False])
    .groupby(['province','system'])
    .head(10)
    [['province','system','Genus','Species','score']])
    topk.to_csv(Path('outputs/tables')/f'top10_species_{cfg.country_code}.csv', index=False)
    print("Saved: outputs/tables/top10_species_*.csv")


if __name__ == "__main__":
    main()
