ADS:
Temporal:
- Line graph, demand, each year
- Bar chart, hours
- Bar chart, days
- hours + days heatmap

 Spatial:
  - chloropleth zones (4/6 times)
  - Journey chloropleth map for each zone (total)
 - Plot zoning clusters on line graph (coloured)

 Event:
 - weather & time series
  - zone and time heatmap
 - Outlier visualisation (box plots for each zone, demand graph with outliers highlighted)

 -animated map in appendix
 - The framing "events" suggests explicitly comparing data w/o outliers with outliers - so thats where a RF comes in
    - outliers in time & location & weather entails outliers in demand
- Do pre-2016 location after.
And then make a time-based model with that data as columns, if h1, otherwise, ignore it and try h0.
MSE lost of both models will support or disagree with hypothesis test results.

https://arxiv.org/pdf/1906.00121
https://github.com/nnzhan/Graph-WaveNet
- Week long residuals
- 15-minute blocks to model taxi propagation
    - do a stock run first
    - use sparse attention. ?
    - dilated convs. vs. temporal attn (+ RoPE)

- Weather causes human behaviour change.
- Based on zoning and location information,
- Then predicts demand


- Data augmentation:
Embeddings:
 sin cos. don't use learnt embeddings unless you want to get fuckd



---
## Taxi Zone Graph
For PDFormer, connections between taxi zones were required.
`generate_adj_grid.py`
First, take the taxi zone shapefile from NYCTLC. Connections calculated through spactially adjacent zones for an adjacency matrix.
This ignored bridges and tunnels, resulting in missing adjacencies.

NYC Graph ML dataset [https://www.kaggle.com/datasets/crailtap/street-network-of-new-york-in-graphml/data] bridges and tunnels extracted, overlapping taxi zones correspond with added edges in the graph. 
`visualise_adj_map` Visualises the graph and taxi zones for verification
<img width="3315" height="3360" alt="image" src="https://github.com/user-attachments/assets/a86c125e-8ce3-4229-9230-2907b3bc1fe4" />



