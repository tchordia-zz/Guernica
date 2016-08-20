require('torch')
require('nn')
require('image')

--make local
m = require('model')
local model = m.mo
local criterion = m.loss

set = require('load')
local trainset = set.trainset
print(trainset:size())
print(model)
print(criterion)
local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.learningRateDecal = .1
trainer.maxIteration = 30 -- do 200 epochs of training
print("start training!")
trainer:train(trainset)
