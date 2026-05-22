suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratWrappers)
  library(rliger)
  library(dplyr)
  library(anndataR)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop(paste(
    "Usage: Rscript batch_correction.R",
    "<input_h5ad> <output_dir> <batch_col> <methods_comma_separated>",
    "<n_hvg> <n_pcs> <liger_k> <params_json>"
  ))
}

input_h5ad <- args[1]
output_dir <- args[2]
batch_col <- args[3]
methods_to_run <- unlist(strsplit(args[4], ",", fixed = TRUE))
n_hvg <- as.integer(args[5])
n_pcs <- as.integer(args[6])
liger_k <- as.integer(args[7])
params_json <- args[8]

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

params <- list()
if (!is.na(params_json) && file.exists(params_json)) {
  params <- jsonlite::fromJSON(params_json, simplifyVector = FALSE)
}

param_value <- function(method, key, default = NULL) {
  if (!method %in% names(params)) {
    return(default)
  }
  if (!key %in% names(params[[method]])) {
    return(default)
  }
  params[[method]][[key]]
}

write_embedding <- function(embedding, method) {
  path <- file.path(output_dir, paste0(method, ".csv"))
  embedding <- as.data.frame(embedding)
  embedding <- tibble::rownames_to_column(embedding, var = "cell_id")
  write.csv(embedding, path, row.names = FALSE)
}

split_assay_by_batch <- function(object, assay_name, batch_col) {
  batch_vector <- object@meta.data[[batch_col]]
  names(batch_vector) <- rownames(object@meta.data)
  object[[assay_name]] <- split(object[[assay_name]], f = batch_vector)
  object
}

message(">>> Loading h5ad as Seurat: ", input_h5ad)
seurat_obj <- anndataR::read_h5ad(input_h5ad, as = "Seurat")
assay_name <- DefaultAssay(seurat_obj)

if (!batch_col %in% colnames(seurat_obj@meta.data)) {
  stop(sprintf("Batch column '%s' not found in Seurat metadata.", batch_col))
}

message(sprintf(
  ">>> Running R batch correction methods on assay '%s' with batch column '%s'",
  assay_name,
  batch_col
))

if ("CCA" %in% methods_to_run) {
  message(">>> [Method] Running CCA Integration...")
  cca_obj <- seurat_obj
  cca_obj <- split_assay_by_batch(cca_obj, assay_name, batch_col)
  cca_obj <- cca_obj %>%
    NormalizeData(verbose = FALSE) %>%
    FindVariableFeatures(nfeatures = n_hvg, verbose = FALSE) %>%
    ScaleData(verbose = FALSE) %>%
    RunPCA(npcs = n_pcs, verbose = FALSE)
  cca_obj <- IntegrateLayers(
    object = cca_obj,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction = "cca",
    verbose = FALSE
  )
  write_embedding(cca_obj@reductions$cca@cell.embeddings, "CCA")
  message("--- CCA integration: OK")
}

if ("RPCA" %in% methods_to_run) {
  message(">>> [Method] Running RPCA Integration...")
  rpca_obj <- seurat_obj
  rpca_obj <- split_assay_by_batch(rpca_obj, assay_name, batch_col)
  rpca_obj <- rpca_obj %>%
    NormalizeData(verbose = FALSE) %>%
    FindVariableFeatures(nfeatures = n_hvg, verbose = FALSE) %>%
    ScaleData(verbose = FALSE) %>%
    RunPCA(npcs = n_pcs, verbose = FALSE)
  rpca_obj <- IntegrateLayers(
    object = rpca_obj,
    method = RPCAIntegration,
    orig.reduction = "pca",
    new.reduction = "rpca",
    verbose = FALSE
  )
  write_embedding(rpca_obj@reductions$rpca@cell.embeddings, "RPCA")
  message("--- RPCA integration: OK")
}

if ("FastMNN" %in% methods_to_run) {
  message(">>> [Method] Running FastMNN Integration...")
  fastmnn_obj <- seurat_obj
  fastmnn_obj <- split_assay_by_batch(fastmnn_obj, assay_name, batch_col)
  fastmnn_obj <- fastmnn_obj %>%
    NormalizeData(verbose = FALSE) %>%
    FindVariableFeatures(nfeatures = n_hvg, verbose = FALSE) %>%
    ScaleData(verbose = FALSE) %>%
    RunPCA(npcs = n_pcs, verbose = FALSE)
  fastmnn_obj <- IntegrateLayers(
    object = fastmnn_obj,
    method = FastMNNIntegration,
    new.reduction = "fastmnn",
    verbose = FALSE
  )
  write_embedding(fastmnn_obj@reductions$fastmnn@cell.embeddings, "FastMNN")
  message("--- FastMNN integration: OK")
}

if ("Liger" %in% methods_to_run || "rliger" %in% methods_to_run) {
  message(">>> [Method] Running Liger Integration...")
  liger_obj <- seurat_obj
  liger_obj <- split_assay_by_batch(liger_obj, assay_name, batch_col)
  method_liger_k <- param_value("Liger", "k", liger_k)
  liger_obj <- liger_obj %>%
    normalize() %>%
    selectGenes() %>%
    scaleNotCenter() %>%
    runINMF(k = as.integer(method_liger_k)) %>%
    quantileNorm()
  write_embedding(liger_obj@reductions$inmfNorm@cell.embeddings, "Liger")
  message("--- Liger integration: OK")
}

message(">>> R batch correction finished.")
