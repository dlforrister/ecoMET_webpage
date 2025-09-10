#' Internal: assert an optional package is installed
#' @keywords internal
#' @noRd
.require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(pkg, " is required for this function. Install it with install.packages('", pkg, "').", call. = FALSE)
  }
}
