
# Plot figure 1: Global wave status and T0 ---------------------------------------------------------------

# Load Packages, Clear, Sink -------------------------------------------------------

# load packages
package_list <- c("readr","ggplot2","gridExtra","plyr","dplyr","ggsci","RColorBrewer",
                  "viridis","sf","reshape2","ggpubr","egg","scales","plotrix","ggallin", "stats")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)

# clear workspace
rm(list=ls())

# disable scientific notation
options(scipen=999)

# Import Data for figure 1 -------------------------------------------------------------------

figure_1a_data <- read_csv(file="./data/figure_1a.csv",
                           na = c("N/A","NA","#N/A"," ",""))
figure_1a_data$countrycode <- as.factor(figure_1a_data$countrycode)

figure_1b_data <- read_delim(file="./data/figure_1b.csv",
                             delim=";",
                             na = c("N/A","NA","#N/A"," ","","None"))
figure_1b_data$countrycode <- as.factor(figure_1b_data$countrycode)
figure_1b_data$class <- as.factor(figure_1b_data$class)
figure_1b_data$class_coarse <- as.factor(figure_1b_data$class_coarse)

# Process Data for figure 1 -------------------------------------------------------------------

# Remove rows with NA in geometry. Required to convert column to shape object
figure_1b_data <- subset(figure_1b_data,!is.na(geometry))
# Convert "geometry" column to a sfc shape column 
figure_1b_data$geometry <- st_as_sfc(figure_1b_data$geometry)
# Convert dataframe to a sf shape object with "geometry" containing the shape information
figure_1b_data <- st_sf(figure_1b_data)

# Figure 1a   ----------------------------------------------
# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- "GnBu"
my_palette_4 <- '#cb181d'

figure_1a1 <- (ggplot(figure_1a_data, aes(x=date, y=new_per_day))
               + geom_line(aes(color=countrycode)
                           , size=0.1, alpha=0.4, na.rm=TRUE, color=my_palette_2, show.legend=FALSE)
               + geom_smooth(method = "loess", se = FALSE, span=0.2, na.rm=TRUE, color=my_palette_4)
               + labs(title="Confirmed Cases per Day", x=element_blank(), y=element_blank())
               + theme_classic(base_size=8,base_family='serif')
               + scale_y_continuous(trans='log', breaks=c(1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000))
               + scale_x_date(date_breaks='3 months', date_labels='%b %Y')
               + theme(plot.title=element_text(size=8, hjust = 0.5), panel.grid.major.y = element_line(size=.1, color="grey")))
#figure_1a1
ggsave('./plots/figure_1a1.png', plot=figure_1a1, width=10, height=7, units='cm', dpi=300)

figure_1a2 <- (ggplot(figure_1a_data, aes(x=date, y=dead_per_day))
               + geom_line(aes(color=countrycode)
                           , size=0.1, alpha=0.4, na.rm=TRUE, color=my_palette_2, show.legend=FALSE)
               + geom_smooth(method = "loess", se = FALSE, span=0.2, na.rm=TRUE, color=my_palette_4)
               + labs(title="Deaths per Day", x=element_blank(), y=element_blank())
               + theme_classic(base_size=8,base_family='serif')
               + scale_y_continuous(trans='log', breaks=c(1,2,5,10,20,50,100,200,500,1000,2000,5000))
               + scale_x_date(date_breaks='3 months', date_labels='%b %Y')
               + theme(plot.title=element_text(size=8, hjust = 0.5), panel.grid.major.y = element_line(size=.1, color="grey")))
#figure_1a2
ggsave('./plots/figure_1a2.png', plot=figure_1a2, width=10, height=7, units='cm', dpi=300)



# Figure 1b1: Chloropleth   ----------------------------------------------
figure_1b1 <- (ggplot(data = figure_1b_data) 
               + geom_sf(aes(fill=days_to_t0_10_dead), lwd=0, color=NA, na.rm=TRUE)
               + labs(title=element_blank(), fill="Days until Epidemic Established")
               + scale_fill_distiller(palette=my_palette_3, trans="sqrt", breaks=c(1,100,200,300,400,500))
               #+ scale_x_continuous(expand=c(0,0), limits=c(-125, -65))
               #+ scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 20, barheight = 0.5))
               + theme(legend.text=element_text(size=8,family='serif'),legend.title=element_text(vjust=1,size=8,family='serif')
                       , panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
#figure_1b1
ggsave('./plots/figure_1b1.png', plot=figure_1b1, width=20, height=12, units='cm', dpi=300)


# Figure 1b2: Boxplot   ----------------------------------------------
figure_1b2 <- (ggplot(data = figure_1b_data) 
               + geom_boxplot(aes(x=class_coarse, group=class_coarse, y=days_to_t0_10_dead)
                              , na.rm=TRUE, outlier.colour=my_palette_2, outlier.shape=1, fill=my_palette_2)
               + labs(title=element_blank(), x=element_blank(), y="Days until Epidemic Established")
               + scale_x_discrete(limits=rev(levels(figure_1b_data$class_coarse)),labels=c('Third Wave or Above','Second Wave','First Wave'))
               + scale_y_continuous(expand=c(0,0), limits=c(0, NA))
               + coord_flip()
               + theme_classic(base_size=8,base_family='serif')
               + theme(panel.grid.major.x = element_line(size=.1, color="grey")))
#figure_1b2
ggsave('./plots/figure_1b2.png', plot=figure_1b2, width=10, height=7, units='cm', dpi=300)


# Figure 1b3: Scatterplot of GNI   ----------------------------------------------
figure_1b3 <- (ggplot(data = figure_1b_data, aes(x=gni_per_capita, y=days_to_t0_10_dead)) 
               + geom_point(na.rm=TRUE, color=my_palette_2, shape=1)
               + geom_smooth(method='lm', color=my_palette_4, se=FALSE)
               + labs(title=element_blank(), x='GNI per Capita', y="Days until Epidemic Established")
               + scale_x_continuous(trans='log',breaks=c(1000,2000,5000,10000,20000,50000))
               + scale_y_continuous(trans='log', breaks=c(100,200,300,400,500))
               + theme_classic(base_size=8,base_family='serif')
               + theme(panel.grid.major = element_line(size=.1, color="grey")))
#figure_1b3
ggsave('./plots/figure_1b3.png', plot=figure_1b3, width=10, height=7, units='cm', dpi=300)


