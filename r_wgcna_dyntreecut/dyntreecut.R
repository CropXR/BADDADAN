library(WGCNA)
library(arrow)
library(yaml)

dynamictree_on_combined_dists <- function(in_path, out_dir, yaml_path) {
  coexp_mat <- read_parquet(in_path)
  coexp_mat <- as.data.frame(coexp_mat)
  
  yaml <- yaml.load_file(yaml_path)
  minModuleSize <- yaml$hyperparams$r_min_module_size
  
  # Set the row names 
  rownames(coexp_mat) <- coexp_mat$`__index_level_0__`
  
  # Remove final column
  coexp_mat <- coexp_mat[,-ncol(coexp_mat)]
  
  gene_tree <- hclust(as.dist(coexp_mat), method='average')

  # Regular tree cut was used for debugging
  # dynamicMods <- cutree(gene_tree, 400)
  deepSplitList <-   yaml$hyperparams$r_deep_split
  # This is list so split it up now
  
  for (deepSplit in deepSplitList){
    
    # Module identification using dynamic tree cut:
    dynamicMods = cutreeDynamic(dendro = gene_tree, distM = coexp_mat,
                                deepSplit = deepSplit, pamRespectsDendro = TRUE,
                                minClusterSize = minModuleSize);
    
    # dynamicColors = labels2colors(dynamicMods)
  
    file_name <- basename(in_path)
  
    # plot_name <- sub("\\.parquet\\.gzip$", "_dendrogram.pdf", file_name)  
    # pdf(file=file.path(out_dir, plot_name))
    # plotDendroAndColors(gene_tree, dynamicColors, "Dynamic Tree Cut",
    #                     dendroLabels = FALSE, hang = 0.03,
    #                     addGuide = TRUE, guideHang = 0.05,
    #                     main = "Gene dendrogram and module colors")
    # dev.off()
    
    module_df <- data.frame(
      gene_id = rownames(coexp_mat),
      colors = dynamicMods
    )
    
    file_name <- sub("\\.parquet\\.gzip$",
                     paste0("_wgcna_clustered_ds", deepSplit, '.csv'),
                     file_name)  
    out_path <- file.path(out_dir, file_name)
    
    write.csv(module_df, out_path)
  }
}

main <- function(in_dir, out_dir){
  files <- list.files(path = in_dir, 
                      pattern = "*.parquet.gzip",
                      full.names = TRUE)
  if (!dir.exists(out_dir)) {dir.create(out_dir)}
  
  yaml_config <- file.path(dirname(dirname(dirname(in_dir))), 'config.yaml')
  
  if (!file.exists(yaml_config)) {
    yaml_config <- file.path(dirname(dirname(in_dir)), 'config.yaml')
  }
  
  mapply(dynamictree_on_combined_dists, 
         files,
         MoreArgs = list(out_dir, yaml_config),
         SIMPLIFY = FALSE)
}

do_drought <- function(){
  message('Processing drought')
  for (word in c('atted', 'local', 'combined_min', 'combined_sum')){
    in_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/drought/jackknifes", word)
    out_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/drought/r_output", word)
    message(paste('Processing', word))
    main(in_dir, out_dir)
  }
  
  main("../data/experiments/18_robustness_with_wgcna_cutting/drought/full_datasets",
       "../data/experiments/18_robustness_with_wgcna_cutting/drought/r_output/full_datasets")
}

do_heat <- function(){
  message('Processing heat')
  for (word in c('atted', 'local', 'combined_min', 'combined_sum')){
    in_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/heat/jackknifes", word)
    out_dir <- file.path("../data/experiments/18_robustness_with_wgcna_cutting/heat/r_output", word)
    message(paste('Processing', word))
    main(in_dir, out_dir)
  }
  
  main("../data/experiments/18_robustness_with_wgcna_cutting/heat/full_datasets",
       "../data/experiments/18_robustness_with_wgcna_cutting/heat/r_output/full_datasets")
}

do_heat()
do_drought()

