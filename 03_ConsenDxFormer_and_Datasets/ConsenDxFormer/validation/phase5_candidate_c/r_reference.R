# Synthetic numeric cross-check only. Never read original model predictions.
# Requires reviewed R runtime, and clust.bin.pair 0.1.2 with the pinned formula.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop('Usage: Rscript r_reference.R INPUT_DIR NEW_OUTPUT_DIR EXPECTED_R_VERSION')
input <- normalizePath(args[1], mustWork=TRUE)
output <- args[2]
if (dir.exists(output) || file.exists(output)) stop('Output exists; refusing overwrite')
if (as.character(getRversion()) != args[3]) stop('R version differs from explicitly pinned runtime')
if (!requireNamespace('clust.bin.pair', quietly=TRUE)) stop('clust.bin.pair is not installed')
if (as.character(utils::packageVersion('clust.bin.pair')) != '0.1.2') stop('Unexpected clust.bin.pair version')
impl <- getFromNamespace('.durkalski.impl', 'clust.bin.pair')
expected.body <- quote(sum( (bk-ck)/nk )^2 / sum( ((bk-ck)/nk)^2 ))
if (!identical(body(impl), expected.body)) stop('Pinned Durkalski formula mismatch')
dir.create(output, recursive=FALSE)
if (!dir.exists(output)) stop('Cannot reserve output directory')
# Pin exact synthetic fixture bytes before reading numerical inputs.
expected.md5 <- c(
  'binomial_cases.csv'='2fb23a0ead1059a79cae2bcc50fe2f63',
  'holm_cases.csv'='6f986352d8fc18160ad14c729665b54c',
  'legacy_counts.csv'='d949b9c5fbad74344b30697f6b2dab4e',
  'legacy_expected.csv'='fea00a4be65dd681e56b3cf55983060b'
)
actual.md5 <- unname(tools::md5sum(file.path(input,names(expected.md5))))
if (!identical(actual.md5, unname(expected.md5))) stop('Synthetic reference fixture integrity mismatch')
read <- function(name) read.csv(file.path(input,name), colClasses='character', check.names=FALSE)
rat <- function(x) {
  vapply(x, function(s) {
    if (is.na(s) || s == '') return(NA_real_)
    p <- strsplit(s,'/',fixed=TRUE)[[1]]
    if (length(p)==1L) return(as.numeric(p))
    if (length(p)!=2L) stop('Invalid rational')
    as.numeric(p[1])/as.numeric(p[2])
  }, numeric(1))
}
close <- function(actual, expected, tol=5e-13) {
  if (is.na(expected)) return(is.na(actual))
  is.finite(actual) && abs(actual-expected) <= tol*max(1,abs(expected))
}
counts <- read('legacy_counts.csv'); ex <- read('legacy_expected.csv')
test <- getFromNamespace('.durkalski.test','clust.bin.pair')
legacy.results <- list()
for (i in seq_len(nrow(ex))) {
  id <- ex$case_id[i]; rows <- counts[counts$case_id==id,]
  q <- test(rat(rows$a),rat(rows$b),rat(rows$c),rat(rows$d))
  p <- if (is.na(q)) NA_real_ else stats::pchisq(q,df=1,lower.tail=FALSE)
  if (!close(q,rat(ex$Q[i])) || !close(p,rat(ex$p[i]))) stop(paste('Legacy mismatch',id))
  legacy.results[[i]] <- data.frame(case_id=id,Q=q,p=p)
}
h <- read('holm_cases.csv'); holm.results <- list()
for (id in unique(h$case_id)) {
  rows <- h[h$case_id==id,]; if (nrow(rows)!=15L || anyDuplicated(rows$id)) stop('Bad family')
  raw <- rat(rows$raw); undefined <- is.na(raw)
  work <- raw;work[undefined] <- 1
  adjusted <- stats::p.adjust(work,method='holm',n=15L)
  adjusted[undefined] <- NA_real_
  expected <- rat(rows$adjusted)
  if (!all(mapply(close,adjusted,expected))) stop(paste('Holm mismatch',id))
  reject <- !undefined & adjusted<=.05
  if (!identical(as.integer(reject),as.integer(rows$reject))) stop(paste('Holm decision mismatch',id))
  holm.results[[id]] <- data.frame(case_id=id,id=rows$id,adjusted=adjusted,reject=reject)
}
b <- read('binomial_cases.csv'); binomial.results <- list()
for (i in seq_len(nrow(b))) {
  result <- stats::binom.test(rat(b$rejections[i]),rat(b$B[i]),conf.level=rat(b$confidence[i]))
  ci <- as.numeric(result$conf.int)
  if (!close(ci[1],rat(b$lower[i])) || !close(ci[2],rat(b$upper[i]))) stop('Clopper-Pearson mismatch')
  binomial.results[[i]] <- data.frame(rejections=rat(b$rejections[i]),lower=ci[1],upper=ci[2])
}
write.csv(do.call(rbind,legacy.results),file.path(output,'legacy_results.csv'),row.names=FALSE)
write.csv(do.call(rbind,holm.results),file.path(output,'holm_results.csv'),row.names=FALSE)
write.csv(do.call(rbind,binomial.results),file.path(output,'binomial_results.csv'),row.names=FALSE)
writeLines(c(capture.output(sessionInfo()),'', 'PINNED CRAN SOURCE: e464ac7e5d094bffd25b7c4cbba67820b60f1cc1',
  'Durkalski source blob: f6523d982dca310c1a76954c9965e36aaa2f9cf1',
  'R stats p.adjust function:',deparse(stats::p.adjust),'R stats binom.test function:',deparse(stats::binom.test)),
  file.path(output,'reference_identity.txt'))
writeLines(c('R_REFERENCE_COMPARISONS=PASS','SYNTHETIC_ONLY=True','RESEARCH_READY=False',
             'Independent review and source-loader integration remain pending.'),file.path(output,'STATUS.txt'))
cat('R_REFERENCE_COMPARISONS=PASS; RESEARCH_READY=False\n')
