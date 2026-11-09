required_packages <- c("ggplot2", "dplyr", "tidyr", "akima")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(akima)

if (!file.exists("simulation_results.csv") || !file.exists("coverage_results.csv")) {
  stop("Python CSV outputs are missing. Run generate_data.py first.")
}

# Read computed data from Python
simulation_df <- read.csv("simulation_results.csv")
coverage_df <- read.csv("coverage_results.csv")

# -------------------------------------------------------------------
# 1. Type I error line plot
# -------------------------------------------------------------------
plot_type1_error <- function(df) {
  df_long <- df %>%
    mutate(sample_size = as.numeric(sample_size), p_baseline = as.numeric(p_baseline))

  p_levels <- sort(unique(df_long$p_baseline))

  plt <- ggplot(df_long, aes(x = sample_size, y = type1_z, group = p_baseline)) +
    geom_line(aes(color = "Z-Test"), linewidth = 1.1) +
    geom_point(aes(color = "Z-Test"), size = 2) +
    geom_line(aes(y = type1_lrt, color = "LRT"), linewidth = 1.1, linetype = "dashed") +
    geom_point(aes(y = type1_lrt, color = "LRT"), size = 2) +
    geom_hline(yintercept = 0.05, color = "red", linetype = "dotted") +
    scale_x_log10() +
    facet_wrap(~ p_baseline, labeller = label_both, ncol = length(p_levels)) +
    labs(
      x = "Sample Size per Group (n)",
      y = "Empirical Type I Error Rate",
      color = "Method",
      title = "Empirical Type I Error Rate vs. Sample Size"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave("type1_error_plot.png", plot = plt, width = 12, height = 5, dpi = 300)
  print(plt)
}

# -------------------------------------------------------------------
# 2. Confidence interval coverage plot
# -------------------------------------------------------------------
plot_coverage <- function(df) {
  df_long <- df %>%
    select(sample_size, p_baseline, coverage_wald, coverage_score) %>%
    pivot_longer(cols = c(coverage_wald, coverage_score), names_to = "method", values_to = "coverage") %>%
    mutate(
      method = recode(method,
        coverage_wald = "Wald CI",
        coverage_score = "Score-style CI"
      )
    )

  plt <- ggplot(df_long, aes(x = sample_size, y = coverage, group = method)) +
    geom_line(aes(color = method), linewidth = 1.1) +
    geom_point(aes(color = method), size = 2) +
    geom_hline(yintercept = 0.95, color = "red", linetype = "dotted") +
    scale_x_log10() +
    facet_wrap(~ p_baseline, labeller = label_both, ncol = length(unique(df_long$p_baseline))) +
    labs(
      x = "Sample Size per Group (n)",
      y = "Empirical Coverage",
      title = "95% Confidence Interval Coverage vs. Sample Size",
      color = "Interval Type"
    ) +
    coord_cartesian(ylim = c(0.75, 1.0)) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave("coverage_plot.png", plot = plt, width = 12, height = 5, dpi = 300)
  print(plt)
}

# -------------------------------------------------------------------
# 3. Smooth contour phase diagram in n vs p space
# -------------------------------------------------------------------
plot_phase_diagram <- function(df, metric = "type1_z") {
  if (!(metric %in% names(df))) {
    stop(sprintf("Metric '%s' not found in the simulation table.", metric))
  }

  plot_df <- df %>%
    rename(n = sample_size, p = p_baseline) %>%
    mutate(
      error_rate = .data[[metric]],
      n_interp = log10(n),
      p_interp = qlogis(p)
    )

  x_vals <- seq(min(plot_df$n_interp), max(plot_df$n_interp), length.out = 300)
  y_vals <- seq(min(plot_df$p_interp), max(plot_df$p_interp), length.out = 300)

  interp <- with(
    plot_df,
    akima::interp(
      x = n_interp,
      y = p_interp,
      z = error_rate,
      xo = x_vals,
      yo = y_vals
    )
  )

  interp_df <- expand.grid(n_interp = interp$x, p_interp = interp$y)
  interp_df$error_rate <- as.vector(interp$z)
  interp_df <- interp_df %>%
    mutate(
      n = 10 ^ n_interp,
      p = plogis(p_interp)
    )

  title_label <- if (metric == "type1_z") {
    "Smooth empirical Type I error contour map (Z-test)"
  } else if (metric == "type1_lrt") {
    "Smooth empirical Type I error contour map (LRT)"
  } else {
    sprintf("Smooth empirical contour map (%s)", metric)
  }

  plt <- ggplot(interp_df, aes(x = n, y = p, z = error_rate)) +
    geom_contour_filled(
      breaks = c(0.04, 0.045, 0.05, 0.06, 0.08, 1.0),
      alpha = 0.9
    ) +
    geom_contour(
      breaks = c(0.04, 0.045, 0.05, 0.06, 0.08),
      color = "black",
      linewidth = 0.4
    ) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "black", alpha = 0.7) +
    scale_x_continuous(trans = "log10") +
    labs(
      x = "Sample size per group, n",
      y = "Baseline conversion rate, p",
      title = title_label
    ) +
    theme_minimal(base_size = 12)

  ggsave(
    filename = paste0("phase_diagram_", metric, ".png"),
    plot = plt,
    width = 8,
    height = 6,
    dpi = 300
  )
  print(plt)
}

# Generate all plots
plot_type1_error(simulation_df)
plot_coverage(coverage_df)
plot_phase_diagram(simulation_df, metric = "type1_z")
plot_phase_diagram(simulation_df, metric = "type1_lrt")

cat("Plots saved to the working directory.\n")
