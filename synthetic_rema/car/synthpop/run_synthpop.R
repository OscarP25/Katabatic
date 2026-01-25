suppressMessages(library(synthpop))
set.seed(42)

data <- read.csv("/content/drive/MyDrive/katabatic1/synthetic/car/synthpop/train_full.csv")

syn_data <- syn(
  data,
  method = "cart",
  seed = 42
)

write.csv(
  syn_data$syn,
  "/content/drive/MyDrive/katabatic1/synthetic/car/synthpop/synthetic_full.csv",
  row.names = FALSE
)