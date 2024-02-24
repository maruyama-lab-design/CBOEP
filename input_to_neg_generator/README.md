EPI datasets used for input of our negative generators and EPI predictors.

| Folder name | Description |
| :---:| ---- |
|unnormalized_BENGI|EPI data from github of TransEPI (https://github.com/biomed-AI/TransEPI/tree/main/data/BENGI). All datasets with a same cell line are mearged. Then, duplicate EP pairs were removed, and duplicate pairs containing a mixture of positive and negative labels were removed, leaving the positive one.|
|normalized_BENGI|From "unnormalized_BENGI", negative EP pairs containing enhancers or promoters that don't occur in the positive EP pairs are removed.|
|TargetFinder|EPI data from github of TargetFinder (https://github.com/shwhalen/targetfinder/tree/master).|

