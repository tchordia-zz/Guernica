require 'nn'
require 'lfs'
require 'image'
require 'csvigo'

-- set constants
local imwidth = 100
local imheight = 100
local numchannels = 3
local numcategories = 10
local limit = 1000
local numsamples = 1000

--set wd
swd = lfs.currentdir() .. '/train_scaled/'
pwd = swd --lfs.currentdir() .. '/train_4/'

print(pwd)

local shouldsave = true;
i = 1
local trainset = {}
local file2index = {}

trainset.data = {} --torch.Tensor(numsamples, numchannels, imwidth, imheight):zero()
-- trainset.labels = torch.Tensor(numsamples, numcategories):zero()

print("done creating Tensors!")
for file in paths.files(pwd) do
    if(file:find('.jpg$') and i < limit) then
        temp_im = image.scale(image.load(pwd .. file), imwidth, imheight)
        if (temp_im:size(1) == 1) then
            print("is grayscale...")
            temp_im = torch.cat(torch.cat(temp_im, temp_im, 1), temp_im, 1)
            print(temp_im.shape())
        end

        if temp_im:size(1) == numchannels and temp_im:size(2) == imheight and temp_im:size(3) == imwidth then
            -- if shouldsave then image.save(swd .. file, temp_im) end
            trainset.data[i] = temp_im
            file2index[file] = i
            i = i + 1
        else
            print("image wrong size: ")
            print(temp_im:size())
        end
    end
end

print("done loading image data!")
print(#trainset.data)

-- Load csv data and create table to access artist/filename etc
labels = csvigo.load({path = "train_info.csv", mode = "large"})
access = {}
for key, value in pairs(labels[1]) do
    access[value] = key
end

local numArtists = 0
local getIndexFromArtist = {}
local index2val = {}

--Get a unique artist id for each artist
for i = 1, #labels do
    local row = labels[i]
    local artistName = row[access.artist]
    local filename = row[access.filename]

    if file2index[filename] ~= nil then
        -- if the artist doesn't have a unique id, assign one
        if getIndexFromArtist[artistName] == nil then
            numArtists = numArtists + 1
            getIndexFromArtist[artistName] = numArtists;
        end
        -- assign each index the id of the artist who wrote it
        index2val[file2index[filename]] = getIndexFromArtist[artistName]
    end
end

final_labels = {}

-- convert each number from index2val into a tensor of size numArtists
for key, value in pairs(index2val) do
    local tens = torch.Tensor(numArtists):zero()
    tens:indexFill(1,torch.LongTensor{value},1)
    final_labels[key] = tens
end

trainset.labels = final_labels

setmetatable(trainset, {
  __index = function(self,i)
    return {self.data[i], self.labels[i]}
  end
})

--The network trainer expects a size function
trainset.size = function(self)
    return numsamples - 1
end

print("Done! sizeis :")
print(numsamples)
print(trainset.size(trainset))

-- print(final_labels[2])
-- print(numArtists)
-- print(authors)
-- print(index2val)

return {trainset = trainset}
