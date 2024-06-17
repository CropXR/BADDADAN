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

# DROUGHT FIRST AND SCREW THE FUNCTIONS I'LL JUST COPY PASTE THE CODE :O
do_drought <- function(){
  drought_path = 'drought_expr_matrix.csv'
  drought_out_path = 'drought_expr_matrix_limma_filtered.csv'
  
  
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
  
  # levels to use for design
  lev <- c('zero_0')
  lev <- c(lev, lapply(1:13, function(i) paste("control", i, sep='_')))
  lev <- c(lev, lapply(1:13, function(i) paste("drought", i, sep='_')))
  
  f <- factor(paste(targets$Condition, targets$Time, sep='_'), levels=lev)
  
  design <- model.matrix(~0+f)
  colnames(design) <- lev
  
  drought_contrasts <- makeContrasts(
    Dif1d = (drought_1 - zero_0)    - (control_1 - zero_0),
    Dif2d = (drought_2 - drought_1) - (control_2 - control_1),
    Dif3d = (drought_3 - drought_2) - (control_3 - control_2),
    Dif4d = (drought_4 - drought_3) - (control_4 - control_3),
    Dif5d = (drought_5 - drought_4) - (control_5 - control_4),
    Dif6d = (drought_6 - drought_5) - (control_6 - control_5),
    Dif7d = (drought_7 - drought_6) - (control_7 - control_6),
    Dif8d = (drought_8 - drought_7) - (control_8 - control_7),
    Dif9d = (drought_9 - drought_8) - (control_9 - control_8),
    Dif10d = (drought_10 - drought_9) - (control_10 - control_9),
    Dif11d = (drought_11 - drought_10) - (control_11 - control_10),
    Dif12d = (drought_12 - drought_11) - (control_12 - control_11),
    Dif13d = (drought_13 - drought_12) - (control_13 - control_12),
    levels=design)
  
  
  fit <- lmFit(df, design)
  
  fit2 <- contrasts.fit(fit, drought_contrasts)
  fit2 <- eBayes(fit2)
  out_table = topTable(fit2, number=nrow(df), p.value=.05)
  
  
  "AT2G22540" %in% rownames(out_table)
  # Great news, the gene they found in the paper (AGL22) is also found here
  
  writeClipboard(paste(rownames(head(out_table, n=50)), collapse = "\n"))
  
  # Test plotting a gene
  gene_series <- t(df['AT3G62950',])
  gene_series <- as.data.frame(gene_series)
  colnames(gene_series) <- 'Expression'
  gene_series <- cbind(gene_series, targets)
  
  
  
  # Create the plot
  p <- ggplot(gene_series, aes(x = Time, y = Expression, color = Condition)) +
    geom_point() +
    labs(title = "Expression Over Time", x = "Time", y = "Expression") +
    theme_minimal()
  
  # Display the plot
  print(p)
  
  # Looks DE indeed
  
  out_df <- df[rownames(out_table),]
  
  original_headers <- colnames(read.csv(drought_path, check.names=FALSE))
  
  colnames(out_df) <- original_headers[-1]
  
  write.csv(out_df, drought_out_path)
  # Cols are samples, rows are genes, values are expression values in csv
}


do_heat <- function(){
  heat_path = 'heat_expr_matrix.csv'
  heat_out_path = 'heat_expr_matrix_limma_filtered.csv'
  heat_target_path = 'heat_sample_metadata.csv'
  df = read.csv(heat_path, header=TRUE, row.names=1)
  df = log2(df)
  
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


do_drought()

do_heat()
