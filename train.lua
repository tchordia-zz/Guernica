require('torch')
require('nn')
require('image')

local m = require('model')
model = m.model
criterion = m.loss

set = require('load')
trainset = set.trainset
print(trainset:size())
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 200 -- do 200 epochs of training
print("start training!")
trainer:train(trainset)
