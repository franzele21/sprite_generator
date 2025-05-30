# Discriminator
Can be used for a GAN-like structure

Will output 0 if the provided image is not a Pokémon sprite, and 1 if it is one.

## Important note
The `Discriminator` seems to work (it can train, and predict good results).

The structure is the same as the Encoder, except it has only a binary output. I think it's a good choice, because, I think (intuitively) those two models should be the same complexity.

Before using this method, the `CSVDataset` class should be modified :
- It should take the CSV file of the Pokémon, and another with random 64*64 images (landscapes, faces, etc...). Those one will take the label 0, while the Pokémon ones will take the label 1
- For the Pokémon frame, a random noise should be applied to some of it. For those sprites with the noise, the label should be 0. By doing this, we will force the discriminator not to accept not good enough Pokémon sprites.
- The distribution of "good" (label=1) and "bad" (label=0) images should be 50/50, and maybe half of the "bad" images should be altered Pokémon sprites.

I don't think the `Discriminator` should be modified, but any good changes are welcome.
