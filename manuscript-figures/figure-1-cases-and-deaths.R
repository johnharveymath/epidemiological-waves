library(magrittr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(scales)

y <- read.csv("data/2020-09-13/fig1c-t0-epi-state.csv") %>%
  filter(class == 4) %>%
  use_series(countrycode)

x <- read.csv("data/2020-09-13/figure_5.csv") %>%
  select(countrycode,date,new_per_day,dead_per_day) %>%
  filter(is.element(el = countrycode, set = y)) %>%
  mutate(date = as.Date(date)) %>%
  melt(id.vars = c("countrycode", "date"))

facet_labels <- c(
  dead_per_day = "Deaths",
  new_per_day = "Confirmed cases"
)

g <- ggplot(data = filter(x, countrycode != "USA"), mapping = aes(x = date, y = value)) +
 geom_line() +
 geom_smooth(method = "loess", span = 0.1) +
  facet_wrap(~variable, scales = "free_y", labeller = labeller(variable = facet_labels)) +
 scale_y_sqrt() +
 scale_x_date(labels = label_date_short()) +
 labs(y = NULL, x = NULL) +
  theme_classic()  +
theme(strip.background = element_blank(),
strip.text = element_text(face = "bold"))

ggsave(filename = "./output/png/figure-1-cases-and-deaths.png")
