library(WGCNA)
library(arrow)

dynamictree_on_combined_dists <- function(in_path, out_dir) {
  coexp_mat <- read_parquet(in_path)
  coexp_mat <- as.data.frame(coexp_mat)
  
  # Remove final column
  coexp_mat <- coexp_mat[,-ncol(coexp_mat)]
  
  # Set the row names as the  column names 
  rownames(coexp_mat) <- colnames(coexp_mat)
  
  gene_tree <- hclust(as.dist(coexp_mat), method='average')

  # We like large modules, so we set the minimum module size relatively high:
  minModuleSize = 20;
  # Module identification using dynamic tree cut:
  dynamicMods = cutreeDynamic(dendro = gene_tree, distM = coexp_mat,
                              deepSplit = 2, pamRespectsDendro = FALSE,
                              minClusterSize = minModuleSize);
  table(dynamicMods)
  
  dynamicColors = labels2colors(dynamicMods)
  table(dynamicColors)
  # Plot the dendrogram and colors underneath
  # sizeGrWindow(8,6)
  plotDendroAndColors(gene_tree, dynamicColors, "Dynamic Tree Cut",
                      dendroLabels = FALSE, hang = 0.03,
                      addGuide = TRUE, guideHang = 0.05,
                      main = "Gene dendrogram and module colors")
  
  
  module_df <- data.frame(
    gene_id = names(coexp_mat),
    colors = dynamicColors
  )
  
  file_name <- basename(in_path)
  file_name <- sub("\\.parquet\\.gzip$", "_wgcna_clustered.csv", file_name)  
  
  out_path <- file.path(out_dir, file_name)
  
  write.csv(module_df, out_path)
}


main <- function(in_dir, out_dir){
  files <- list.files(path = in_dir, 
                      pattern = "*.parquet.gzip",
                      full.names = TRUE)
  mapply(dynamictree_on_combined_dists, 
         files,
         MoreArgs = list(out_dir),
         SIMPLIFY = FALSE)
}

for (word in c('atted', 'local', 'combined_min')){
  in_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/drought/jackknifes", word)
  out_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/drought/r_output", word)
  message(paste('Processing', word))
  main(in_dir, out_dir)
}
