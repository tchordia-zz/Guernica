require('torch')
require('nn')
require('image')
local load = require('load')

print("building the model")
local nfeats = 3
local height = 128
local width = height --assume square
local num_classes = load.num_classes

local nstates = {16, 32, 64}
local filtsize = {5,3,3}
local poolsize = {4,4,2}

local pad = function(filter_size)
   local actual_pad = (filter_size - 1)/2
   return nn.SpatialZeroPadding(actual_pad, actual_pad, actual_pad, actual_pad)
  end

local pool_by = function(pf)
   return nn.SpatialMaxPooling(pf,pf,pf,pf)
end
-- maybe try updating with 0 padding
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')
local CNN = nn.Sequential()

-- stage 1: conv+max
--3 x height x width
CNN:add(pad(filtsize[1]))
CNN:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1])) -- nstates[1]x96x96
CNN:add(nn.Threshold())
CNN:add(pool_by(poolsize[1])) --nstates[1]x24x24

-- stage 2: conv+max
CNN:add(pad(filtsize[2]))
CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2])) --nstates[2]x24x24
CNN:add(nn.Threshold())
CNN:add(pool_by(poolsize[2])) --nstates[1]x6x6

--stage 3
CNN:add(pad(filtsize[3]))
CNN:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize[3], filtsize[3])) --nstates[3]x12x12
CNN:add(nn.Threshold())
CNN:add(pool_by(poolsize[3]))  --nstates[3]x3x3

local classifier = nn.Sequential()

local fcc = nstates[3] * width * width / math.pow(poolsize[1] * poolsize[2] * poolsize[3],2)
-- stage 3: linear
classifier:add(nn.Reshape(fcc))
print("no error yet!")
classifier:add(nn.Linear(fcc, num_classes))
print("added linear!")

-- stage 4 : log probabilities
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
loss = nn.ClassNLLCriterion()
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

-- return package:
return {
   mo = model,
   loss = loss,
}
