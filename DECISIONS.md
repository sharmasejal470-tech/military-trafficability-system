# DECISIONS.md

| Decision | Reason |
|---|---|
| Project directory set to `C:\Users\sahil\.gemini\antigravity\scratch\trafficability` | Complies with workspace sandbox directives when no active workspace is bound. |
| Pinned exact verified library versions in `requirements.txt` | Ensures zero-dependency drift and pure-pip compatibility on Streamlit Community Cloud. |
| Base geographic lat/lon (26.9124, 75.7873) is centered on grid with cosine latitude projection | Accurately georeferences the 10x10 km simulation area over Sapta Shakti Command HQ in Jaipur, Rajasthan. |
| Coordinate orientation: row 0 = North, col 0 = West, lat decreases with row, lon increases with col | Standard raster GIS convention aligned with Folium ImageOverlay bounding boxes [[south, west], [north, east]]. |
| Public API provides `(x=col, y=row)` while internal matrices use `[row, col]` | Satisfies coordinate encapsulation rules with clean translation at UI and test boundaries. |
| River depth profile incorporates a natural sandbar/ford zone (< 0.52m nominal depth) | Allows light reconnaissance vehicles to find valid crossings during nominal weather, avoiding an impassable river barrier. |
| River crossing approaches at ford zones (rows 45-53 & 20-30) use stable gravel/loam ramps | Aligns with military river crossing doctrine (FM 90-13) ensuring firm shores for vehicle ingress and egress without marsh entrapment. |
| Monotonic rainfall hazard scaling swells water depth by 15% and degrades substrate firmness by 30% | Guarantees mathematically that rainfall 2.0 never makes any cell safer than at rainfall 1.0. |
| Strict Red-over-Yellow priority mask evaluation in NumPy vectorized pipeline | Prevents marginal warning states from masking fatal immobilizing obstacles (rollover slope, hydrolock, or shear sinkage). |
| Octile distance metric used for 8-neighbor A* pathfinding | Admissible and consistent metric for 8-connected grid movement with diagonal cost sqrt(2). |
| Trafficability overlay rendered as direct PNG buffer from NumPy RGBA array | Eliminates vector polygon DOM bloat in Folium/Leaflet and ensures instant rendering of 10,000 cells. |
| AUV sonar CSV ingestion implemented via `scipy.spatial.cKDTree` nearest-neighbor interpolation | Eliminates C-extension GIS dependencies (GDAL/rasterio) while enabling real-world hydrographic field surveys to override synthetic baselines. |
