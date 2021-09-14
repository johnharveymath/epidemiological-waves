library(dplyr)
library(purrr)
library(magrittr)
library(scales)

x <- read.csv("data/2020-09-15/gni_data.csv",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, gni_per_capita)

y <- read.csv("data/2020-09-13/fig1a-t0-days.csv",
  stringsAsFactors = FALSE
) %>%
  select(countrycode, days_to_t0)


z <- full_join(x = x, y = y, by = "countrycode") %>%
  filter(
    not(is.na(gni_per_capita)),
    not(is.na(days_to_t0))
  )

g <- ggplot(
  data = z,
  mapping = aes(
    x = gni_per_capita,
    y = days_to_t0
  )
) +
  geom_point(shape = 1) +
  geom_smooth(method = "lm") +
  scale_x_log10(
    labels = scales::comma_format(big.mark = ",")
  ) +
  scale_y_log10() +
  labs(
    x = "GNI per capita",
    y = "Days until epidemic established"
  ) +
  theme_classic() +
  theme(axis.title = element_text(face = "bold"))

ggsave(filename = "./output/png/figure-1-t0-and-gni.png")

sink("./output/txt/figure-1-t0-and-gni.txt")
summary(lm(log(days_to_t0) ~ log(gni_per_capita), z))
cat("And again removing the outlier\n")
summary(lm(log(days_to_t0) ~ log(gni_per_capita), filter(z, days_to_t0 > 30)))
sink()
