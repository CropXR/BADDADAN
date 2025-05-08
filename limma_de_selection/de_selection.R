# 
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# if (!require("limma", quietly = TRUE))
#   BiocManager::install("limma")
# 
# if (!require("statmod", quietly = TRUE))
#   install.packages("statmod")

library(limma)
library(splines)
library(ggplot2)
library(dplyr)
library(WGCNA)

# Load required library
# install.packages("VennDiagram")
library(VennDiagram)

do_drought <- function(drought_path, drought_out_path, time_points){
  df = read.csv(drought_path, header=TRUE, row.names=1)
  
  # Do some limma checks to see if samples are alright
  plotMA(df)
  plotMDS(df)
  
  cols = colnames(df)

  split_data <- strsplit(cols, "\\.")
  targets <- do.call(rbind, split_data)
  targets <- as.data.frame(targets, stringsAsFactors = FALSE)
  colnames(targets) <- c("Time", "Condition", "Replicate")
  targets$Time <- as.numeric(gsub("X", "", targets$Time))
  
  # Filter for selected time points
  targets <- targets[targets$Time %in% time_points, ]
  
  # Create a regex pattern to match any of the selected time points
  pattern <- paste0("^X(", paste(time_points, collapse = "|"), ")\\.")
  selected_cols <- colnames(df)[grepl(pattern, colnames(df))]
  df <- df[, selected_cols]
  
  # levels to use for design
  lev <- c('zero_0')
  lev <- c(lev, lapply(time_points[time_points > 0], function(i) paste("control", i, sep='_')))
  lev <- c(lev, lapply(time_points[time_points > 0], function(i) paste("drought", i, sep='_')))
  
  f <- factor(paste(targets$Condition, targets$Time, sep='_'), levels=lev)
  
  design <- model.matrix(~0+f)
  colnames(design) <- lev
  # 
  # drought_contrasts <- makeContrasts(
  #   Dif1d = (drought_1 - zero_0)    - (control_1 - zero_0),
  #   Dif2d = (drought_2 - drought_1) - (control_2 - control_1),
  #   Dif3d = (drought_3 - drought_2) - (control_3 - control_2),
  #   Dif4d = (drought_4 - drought_3) - (control_4 - control_3),
  #   Dif5d = (drought_5 - drought_4) - (control_5 - control_4),
  #   Dif6d = (drought_6 - drought_5) - (control_6 - control_5),
  #   Dif7d = (drought_7 - drought_6) - (control_7 - control_6),
  #   Dif8d = (drought_8 - drought_7) - (control_8 - control_7),
  #   Dif9d = (drought_9 - drought_8) - (control_9 - control_8),
  #   Dif10d = (drought_10 - drought_9) - (control_10 - control_9),
  #   Dif11d = (drought_11 - drought_10) - (control_11 - control_10),
  #   Dif12d = (drought_12 - drought_11) - (control_12 - control_11),
  #   Dif13d = (drought_13 - drought_12) - (control_13 - control_12),
  #   levels=design)
  
  contrast_list <- setNames(
    lapply(2:length(time_points), function(i) {
      if (i == 2) {
        # First contrast should be against zero_0
        paste0(
          "(drought_", time_points[i], " - zero_0) - ",
          "(control_", time_points[i], " - zero_0)"
        )
      } else {
        # The rest follow the usual pattern
        paste0(
          "(drought_", time_points[i], " - drought_", time_points[i - 1], ") - ",
          "(control_", time_points[i], " - control_", time_points[i - 1], ")"
        )
      }
    }),
    paste0("Dif", time_points[-1], "d")
  )
  # Convert to formula-like expression
  drought_contrasts <- makeContrasts(contrasts = unlist(contrast_list), levels = design)
  
  fit <- lmFit(df, design)
  
  fit2 <- contrasts.fit(fit, drought_contrasts)
  fit2 <- eBayes(fit2)
  out_table = topTable(fit2, number=nrow(df), p.value=.05)
  
  out_df <- df[rownames(out_table),]
  # Extract time points and conditions from column names
  colnames(out_df) <- gsub("^X", "", colnames(out_df))  # Remove the "X" if needed

  # Use a regular expression to group by condition and timepoint
  condition_time <- sub("([0-9]+)\\.(control|zero|drought)\\.([a-d])", "\\1_\\2", colnames(out_df))
  
  # Assign the new grouping as the column names
  colnames(out_df) <- condition_time
  
  # Now calculate the mean across replicates (e.g. control_a, control_b, ...)
  out_df_mean <- as.data.frame(sapply(split.default(out_df, sub(" [a-z]$", "", names(out_df))), rowMeans, na.rm = TRUE))
  
  cor_mat <- cor(t(out_df_mean), method = "pearson")  
  dist_mat <- as.dist(1 - cor_mat) 
  gene_tree <- hclust(dist_mat, method = "average")  
  
  dynamicMods = cutreeDynamic(dendro = gene_tree, distM = as.matrix(dist_mat),
                              deepSplit = 1, pamRespectsDendro = TRUE,
                              minClusterSize = 20);
  
  module_df <- data.frame(
    gene_id = rownames(out_df),
    colors = dynamicMods
  )
  
  # Create a string of all selected time points, joined by underscores
  time_point_str <- paste(time_points, collapse = "_")
  
  # Modify the filename to include the time points
  file_name <- paste('drought', time_point_str, "days.csv", sep = "_")
  
  dir.create(file.path(drought_out_path), showWarnings = FALSE)
  
  out_path <- file.path(drought_out_path, file_name)
  
  write.csv(module_df, out_path)
  # Cols are samples, rows are genes, values are expression values in csv
}

do_drought_spline <- function(drought_path, drought_out_path){
  df = read.csv(drought_path, header=TRUE, row.names=1)
  
  # Do some limma checks to see if samples are alright
  plotMA(df)
  plotMDS(df)
  
  # Identify the column(s) with condition 'zero'
  zero_cols <- grep("zero", colnames(df), value = TRUE)
  
  # Duplicate the identified column(s) and rename them
  for (col in zero_cols) {
    new_control_col <- sub("zero", "control", col)
    new_drought_col <- sub("zero", "drought", col)
    
    # Add the new columns to the dataframe
    df[[new_control_col]] <- df[[col]]
    df[[new_drought_col]] <- df[[col]]
  }
  
  # Remove the columns with 'zero' in their names
  df <- df[, !colnames(df) %in% zero_cols]
  
  cols = colnames(df)
  split_data <- strsplit(cols, "\\.")
  targets <- do.call(rbind, split_data)
  targets <- as.data.frame(targets, stringsAsFactors = FALSE)
  colnames(targets) <- c("Time", "Condition", "Replicate")
  targets$Time <- as.numeric(gsub("X", "", targets$Time))
  
  nat_spline <- ns(targets$Time, df=5)
  
  group <- factor(targets$Condition)
  design <- model.matrix(~group*nat_spline)
  
  fit <- lmFit(df, design)
  fit <- eBayes(fit)
  
  out_table = topTable(fit, coef=8:12, number=nrow(df), p.value=.05)
  
  
  # Save data for using in python again
  out_df <- read.csv(drought_path, header=TRUE, row.names=1, check.names = FALSE)
  out_df <- out_df[rownames(out_table),]
  
  write.csv(out_df, drought_out_path)
  # Do some limma checks to see if samples are alright
  plotMA(out_df)
  plotMDS(out_df)
  
  
  # Test plotting a gene
  gene_series <- t(df['AT1G55760',])
  gene_series <- as.data.frame(gene_series)
  colnames(gene_series) <- 'Expression'
  gene_series <- cbind(gene_series, targets)
  
  # Create the plot
  p <- ggplot(gene_series, aes(x = Time, y = Expression, color = Condition)) +
    geom_point() +
    labs(title = "Expression Over Time", x = "Time", y = "Expression") +
    theme_minimal()
  print(p)
  
  
  # Now plot the spline
  investigate_series <- gene_series[targets$'Condition' == 'control',]
  nat_spline <- ns(targets[targets$'Condition' == 'control',]$Time, df=5)
  spline_model <- lm(Expression~nat_spline, data=investigate_series)
  plot(Expression~Time, data=investigate_series)
  points(investigate_series$Time, predict(spline_model), col='red')
  
  investigate_series <- gene_series[targets$'Condition' == 'drought',]
  nat_spline <- ns(targets[targets$'Condition' == 'drought',]$Time, df=5)
  spline_model <- lm(Expression~nat_spline, data=investigate_series)
  plot(Expression~Time, data=investigate_series)
  points(investigate_series$Time, predict(spline_model), col='red')
}


do_heat <- function(heat_path, heat_out_path, heat_target_path){
  df = read.csv(heat_path, header=TRUE, row.names=1)

  plotMDS(df, cex=.5)
  
  # Read sample annotations
  heat_targets_df <- read.csv(heat_target_path)
  
  # Because first column is t=0 for both heat and control, duplicate in df
  df <- data.frame(df[, 1], df)
  names(df)[1] <- paste0("Col", 1, "_copy") # Rename the new column
  
  # And in targets
  heat_targets_df <- rbind(heat_targets_df[1,], heat_targets_df)
  heat_targets_df[1,"Condition"] <- "heat"
  heat_targets_df[2,"Condition"] <- "control"
  
  nat_spline <- ns(heat_targets_df$Time, df=5)

  group <- factor(heat_targets_df$Condition)
  design <- model.matrix(~group*nat_spline)
  
  fit <- lmFit(df, design)
  fit <- eBayes(fit)
  
  out_table = topTable(fit, coef=8:12, number=nrow(df), p.value=.05)
  
  # Remove first column again
  out_df <- df[rownames(out_table),-1]
  
  
  original_headers <- colnames(read.csv(heat_path, check.names=FALSE))
  
  colnames(out_df) <- original_headers[-1]
  # Reuse old column names
  write.csv(out_df, heat_out_path)
  
  # AT5G52640 hsp90
  "AT5G52640" %in% rownames(out_table)
  
  # Test plotting a gene
  gene_series <- t(df['AT5G52640',])
  gene_series <- as.data.frame(gene_series)
  colnames(gene_series) <- 'Expression'
  gene_series <- cbind(gene_series, heat_targets_df)
  
  # Create the plot
  p <- ggplot(gene_series, aes(x = Time, y = Expression, color = Condition)) +
    geom_point() +
    labs(title = "Expression Over Time", x = "Time", y = "Expression") +
    theme_minimal()
  print(p)
  
  
  # coeficients in fit object
  fit$coefficients['AT5G52640']
  
  
  # Now plot the spline man
  investigate_series <- gene_series[heat_targets_df$'Condition' == 'control',]
  nat_spline <- ns(heat_targets_df[heat_targets_df$'Condition' == 'control',]$Time, df=5)
  spline_model <- lm(Expression~nat_spline, data=investigate_series)
  plot(Expression~Time, data=investigate_series)
  points(investigate_series$Time, predict(spline_model), col='red')
  
  investigate_series <- gene_series[heat_targets_df$'Condition' == 'heat',]
  nat_spline <- ns(heat_targets_df[heat_targets_df$'Condition' == 'heat',]$Time, df=5)
  spline_model <- lm(Expression~nat_spline, data=investigate_series)
  plot(Expression~Time, data=investigate_series)
  points(investigate_series$Time, predict(spline_model), col='red')
  
  
  
  ## NOT USED
  # 
  # # Possible names
  # lev <- paste(heat_targets_df$Condition, heat_targets_df$Time, sep='_')
  # 
  # f <- factor(paste(heat_targets_df$Condition, heat_targets_df$Time, sep='_'),
  #             levels = lev)
  # design <- model.matrix(~0+f)
  # colnames(design) <- lev
  # 
  # heat_contrasts <- makeContrasts(
  #   (heat_5 - zero_0) - (control_5 - zero_0),
  #   (heat_10 - heat_5) - (control_10 - control_5),
  #   (heat_20 - heat_10) - (control_20 - control_10),
  #   (heat_40 - heat_20) - (control_40 - control_20),
  #   (heat_60 - heat_40) - (control_60 - control_40),
  #   (heat_80 - heat_60) - (control_80 - control_60),
  #   (heat_100 - heat_80) - (control_100 - control_80),
  #   (heat_120 - heat_100) - (control_120 - control_100),
  #   (heat_140 - heat_120) - (control_140 - control_120),
  #   (heat_160 - heat_140) - (control_160 - control_140),
  #   (heat_180 - heat_160) - (control_180 - control_160),
  #   (heat_200 - heat_180) - (control_200 - control_180),
  #   (heat_220 - heat_200) - (control_220 - control_200),
  #   (heat_240 - heat_220) - (control_240 - control_220),
  #   (heat_260 - heat_240) - (control_260 - control_240),
  #   (heat_280 - heat_260) - (control_280 - control_260),
  #   (heat_300 - heat_280) - (control_300 - control_280),
  #   (heat_320 - heat_300) - (control_320 - control_300),
  #   (heat_340 - heat_320) - (control_340 - control_320),
  #   (heat_360 - heat_340) - (control_360 - control_340),
  #   (heat_640 - heat_360) - (control_640 - control_360),
  #   (heat_1280 - heat_640) - (control_1280 - control_640)
  # , levels = lev)
  # 
  # fit <- lmFit(df, design)
  # 
  # fit2 <- contrasts.fit(fit, heat_contrasts)
  # 
  # fit2 <- eBayes(fit2)
  # topTable(fit2, number=nrow(df), p.value=.05)
  
}

compare_spline_vs_normal_de_drought <- function(drought_out_path,
                                                drought_out_spline_path){
  # Read the two CSV files
  df1 <- read.csv(drought_out_path, row.names = 1)
  df2 <- read.csv(drought_out_spline_path, row.names = 1)
  
  # Get the row names of both dataframes
  rows_df1 <- rownames(df1)
  rows_df2 <- rownames(df2)
  
  # Find overlapping and unique row names
  overlap <- intersect(rows_df1, rows_df2)
  unique_df1 <- setdiff(rows_df1, rows_df2)
  unique_df2 <- setdiff(rows_df2, rows_df1)
  
  # Print the results
  cat("Number of overlapping row names: ", length(overlap), "\n")
  cat("Number of unique row names in pairwise DE: ", length(unique_df1), "\n")
  cat("Number of unique row names in spline DE: ", length(unique_df2), "\n")
  
  # Optional: Create a Venn Diagram
  venn.plot <- venn.diagram(
    x = list(file1 = rows_df1, file2 = rows_df2),
    category.names = c("pairwise DE", "spline DE"),
    filename = NULL
  )
  
  # Display the Venn Diagram
  grid.newpage()
  grid.draw(venn.plot)
}

setwd('C:/Users/noord087/PycharmProjects/d3c2_project/data/experiments/25_everything_including_limma')

# Define the time points sets
time_points_sets <- list(
  0:13,
  c(0, 1, 3, 5, 7, 9, 11, 13),
  c(0, 1, 4, 7, 13)
)

# Loop through each set of time points
for (time_points in time_points_sets) {
  # Print a message indicating which set is running
  message("Running for time points: ", paste(time_points, collapse = ", "))
  
  do_drought('drought/01_input_for_limma.csv', 
             'drought/sparse_sample_experiment/', 
             time_points = time_points)
}

# do_heat('heat/01_input_for_limma.csv', 
#         'heat/sparse_sample_experiment/02_heat_expr_matrix_limma_filtered.csv',
#         '../../raw_data/expression_datasets/emtab375/heat_sample_metadata.csv')

# do_drought_spline('drought/01_input_for_limma.csv',
#                   'drought/sparse_sample_experiment/02b_drought_expr_matrix_limma_spline_filtered.csv')

# compare_spline_vs_normal_de_drought('drought/02a_drought_expr_matrix_limma_filtered.csv',
#                                     'drought/02b_drought_expr_matrix_limma_spline_filtered.csv')
