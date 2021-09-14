library(dplyr)
library(purrr)
library(magrittr)
library(ggplot2)
library(scales)
library(GADMTools)
library(geojsonio)


## read in the t_0 data to colour the regions with.
t0_df <- read.csv(
"data/2020-09-13/fig1a-t0-days.csv",
stringsAsFactors = FALSE
) %>% rename(GID_0 = countrycode)


## read in the geometry of each region and link it to the data via the GID_0.
world_sf <- topojson_read("data/2020-09-13/gadm36_0.json")
plot_sf <- left_join(world_sf, t0_df, by = "GID_0")

## choose some appropriate values for the colour scale
t0_range <- range(purrr::discard(.x = plot_sf$days_to_t0, .p = is.na))
my_breaks <- seq(from = t0_range[1], to = t0_range[2], length = 4)
colours_hex <- c("#5e3c99",
                 "#b2abd2",
                 "#fdb863",
                 "#e66101")

## make the actual plot and do some preliminary styling.
g <- ggplot() +
  geom_sf(data = plot_sf,
          mapping = aes(fill = days_to_t0),
colour = "white",
size = 0.1) +
  scale_fill_gradientn(breaks = my_breaks,
                       colors = colours_hex,
                       limits = range(my_breaks)) +
  labs(fill = "Days until epidemic\nthreshold reached") +
  theme_void() +
  theme(
legend.box.background = element_rect(fill = "white", colour = "grey"),
legend.box.margin = margin(t = 0.1, r = 0.2, b = 0.1, l = 0.2, unit = "cm"),
legend.position = c(0.25, 0.15),
legend.direction = "horizontal",
legend.title = element_text(vjust = 1.0, size = 7),
legend.key.height = unit(0.5, "line"),
legend.text = element_text(size = 5)
)

ggsave(filename = "./output/png/figure-1-t0-map.png",
       plot = g,
height = 10,
width = 17,
units = "cm"
)
