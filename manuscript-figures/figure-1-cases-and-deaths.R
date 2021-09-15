library(magrittr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(scales)
library(zoo)

y_chr <- read.table(
  file = "data/2021-09-15/figure_1b.csv",
  header = TRUE,
  sep = ";",
  stringsAsFactors = FALSE
) %>%
  filter(class == 4) %>%
  use_series(countrycode)

x <- read.csv("data/2021-09-15/figure_1a.csv") %>%
  select(countrycode, date, new_per_day, dead_per_day) %>%
  filter(is.element(el = countrycode, set = y_chr)) %>%
  mutate(date = as.Date(date)) %>%
  melt(id.vars = c("countrycode", "date"))

facet_labels <- c(
  dead_per_day = "Deaths",
  new_per_day = "Confirmed cases"
)

smooth_df_npd <- x %>%
  filter(variable == "new_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(date) %>%
  summarise(mean_val = mean(value)) %>%
  mutate(value = rollmedian(mean_val, 7, na.pad = TRUE), variable = "new_per_day")

smooth_df_dpd <- x %>%
  filter(variable == "dead_per_day", countrycode != "USA", countrycode != "IND") %>%
  group_by(date) %>%
  summarise(mean_val = mean(value)) %>%
  mutate(value = rollmedian(mean_val, 7, na.pad = TRUE), variable = "dead_per_day")

smooth_df <- rbind(smooth_df_npd, smooth_df_dpd)

g <- ggplot(
  data = filter(x, countrycode != "USA", countrycode != "IND"),
  mapping = aes(x = date, y = value, group = countrycode)
) +
  geom_point(shape = 1) +
  geom_line(data = smooth_df, group = NA, colour = "#7a0177", size = 2) +
  facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = facet_labels)) +
  scale_y_sqrt() +
  scale_x_date(labels = label_date_short()) +
  labs(y = NULL, x = NULL) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )

if (interactive()) {
  plot(g)
} else {
  ## Save this to 50% height and 70% width of a landscape A5 page.
  ggsave(
    filename = "./output/png/figure-1-cases-and-deaths.png",
    plot = g,
    height = 0.5 * 14.8,
    width = 0.7 * 21.0,
    units = "cm"
  )
}
