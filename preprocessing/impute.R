library(mice)
merged <- read.csv('merged.csv', header=TRUE, sep=',')
merged.imp <- mice(merged)
write.csv(merged.imp, 'merged_miced.csv')        
