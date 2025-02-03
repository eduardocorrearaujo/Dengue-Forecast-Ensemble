#data <- data.frame(
#  date = rep(seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "month"), 27),
#  UF = rep(state.abb[1:27], each = 12),
#  value = runif(324, 10, 100)
#)
library(geofacet)
library(ggplot2)
library(ggtext)

setwd("/Users/eduardoaraujo/Documents/Github/Dengue-Forecast-Ensemble/paper")

data <- read.csv("../data/dengue_uf.csv.gz")
data$date <- as.Date(data$date)
# Plot
ggplot(data, aes(x = date, y = casos, group = uf)) +
  geom_line() +
  facet_geo(~uf, grid = "br_states_grid1", scales = "free_y") +
  theme_minimal() +
  labs(
    title = "New cases by state (UF)",
    x = "Date",
    y = "New Cases"
  ) +
  
  theme_light() +
  theme(
    plot.title = element_text(size = 22),
    plot.margin = margin(t = 10, r = 20, b = 10, l = 10), # Increase margins
    legend.title = element_blank(), 
    legend.position = "bottom",
    legend.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    strip.text.x = element_text(size = 12, colour = "black"),
    strip.background = element_rect(fill = "gray80")
  )

ggsave('../figures/cases_dengue.pdf', plot = last_plot(), width=18, height=15, dpi =300)

#theme(
#    strip.text = element_textbox_simple(
#     color = "black",        # Text color
#      fill = "gray80",        # Background color (gray rectangle)
#     box.color = "gray60",   # Border color
#      size = 10,              # Font size
#      padding = margin(1, 1, 1, 1), # Padding inside the box
#      margin = margin(1, 1, 1, 1)
#     ))  # Margin around the title box
#      hjust = 1, 
#   
#    ))

