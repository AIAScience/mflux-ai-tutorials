# MFlux.ai tutorials

Open source data science tutorials for [MFlux.ai](https://www.mflux.ai/)

## Contribution guidelines

Contributions are welcome! You can either improve existing tutorials or create new ones.

Each tutorial resides in its own folder. The folder name is the "slug" that defines the last part of the URL where the tutorial will be published on the MFlux website. The folder name should consist of only ASCII letters, numbers, hyphen and underscore characters. Tutorial folders contain `content.md` (the tutorial content in markdown format) and `metadata.json` (some more information about the tutorial). Here is an example of a valid `metadata.json` file:

```json
{
  "title": "Document classification",
  "short_description": "In this tutorial, we will create a simple classifier model that can input video metadata and output a category prediction.",
  "tags": ["deep learning", "keras", "classification"],
  "rank": 0,
  "is_listed": true,
  "cover_image_filename": "cover.png"
}
```

Tutorials should describe a useful case, ideally with some real data instead of toy data. It should be relatively easy to follow the tutorial.

Cover images (`cover.png`) should be 374x260
