from Read.Read_zenodo import open_record

fs=250
#Files=[str(i) for i in range(1,80)]
Files=[str(i) for i in range(2,25)]+[str(i) for i in range(26,62)]+[str(i) for i in range(63,80)]

mode="all_channels"

if mode=="BNC_config":
    Channels=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[20,21,22]] #mode = BNC_config
else:
    Channels=[[i] for i in range(18)]                                              #mode = all_channels

output=open_record(Files[1],mode=mode)
