library(mice)
merged <- read.csv('to_mice.csv', header=TRUE, sep=',')
merged.imp <- mice(merged)
write.csv(merged.imp, 'miced.csv')        
