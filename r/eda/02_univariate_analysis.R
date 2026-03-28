library(readr)
library(dplyr)
library(ggplot2)

input_path <- "data/processed/region_month_base.csv"
plots_path <- "R/outputs/plots/"
tables_path <- "R/outputs/tables/"

dir.create(plots_path, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_path, recursive = TRUE, showWarnings = FALSE)

fire_df <- read_csv(input_path, show_col_types = FALSE)

categorical_vars <- c("Region", "Season")

for (var in categorical_vars) {
  
  # Frequency table
  freq_table <- fire_df %>%
    count(.data[[var]], name = "count") %>%
    mutate(percent = round(count / sum(count) * 100, 2)) %>%
    arrange(desc(count))
  
  write_csv(freq_table, paste0(tables_path, "freq_", var, ".csv"))
  
  print(freq_table)
  
  # bar plot
  p <- ggplot(freq_table, aes(x = .data[[var]], y = count)) +
    geom_col(fill = "steelblue") +
    geom_text(aes(label = count), vjust = -0.4, size = 4) +
    labs(
      title = paste("Distribution of", var),
      x = var,
      y = "Count"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 20, hjust = 1),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank()
    )
  
  ggsave(
    filename = paste0(plots_path, "bar_", var, ".png"),
    plot = p,
    width = 8,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}

# Log-scale histograms (for fire variables)
log_vars <- c(
  "total_burned_area",
  "average_fire_size",
  "worst_case_fire_size"
)

for (var in log_vars) {
  p <- ggplot(fire_df, aes(x = log1p(.data[[var]]))) +
    geom_histogram(bins = 30, fill = "steelblue") +
    labs(
      title = paste("Log-Scaled Histogram of", var),
      x = paste("log(1 +", var, ")"),
      y = "Count"
    ) +
    theme_minimal(base_size = 13)
  
  ggsave(
    filename = paste0(plots_path, "log_hist_", var, ".png"),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300,
    bg = "white"
  )
}