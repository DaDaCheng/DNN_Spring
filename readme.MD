
# A spring–block theory of feature learning in deep neural networks
This repository contains the source code and videos for the paper: ___A spring–block theory of feature learning in deep neural networks___. More details can be found on [arXiv](https://arxiv.org/abs/XXXX.XXXXX).

<div style="text-align: center;">
    <img src="figs/DNN_spring.png" alt="illustration">
</div>


## Vidoes of the folding ruler expermenets
When **slow** and **steady** pulling:

https://private-user-images.githubusercontent.com/13567142/352864542-8fd9e33a-61ed-472a-a14b-884486cff551.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjIxOTc1MzIsIm5iZiI6MTcyMjE5NzIzMiwicGF0aCI6Ii8xMzU2NzE0Mi8zNTI4NjQ1NDItOGZkOWUzM2EtNjFlZC00NzJhLWExNGItODg0NDg2Y2ZmNTUxLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA3MjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNzI4VDIwMDcxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIwYWZlYWNiNDA5MzczMDI1YjdjNGUwYmQ0NDY1OWM2ZDM4ZTNkNGVmNjlkNzcyYTAxNzVlMzg0MTFjZmE2Y2QmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.wGORRXnrPgF6pzcXGpoeRI1HxbbErvj_5jczPKuEMHM

and the corresponding **lazy** training of a DNN on the MNIST dataset:
<div style="text-align: center;">
    <video width="640" height="480" controls>
        <source src="videos/DNN_lazy.mp4" type="video/mp4">
    </video>
</div>

--
When **quick** and **jerky** pulling:
<div style="text-align: center;">
    <video width="640" height="480" controls>
        <source src="videos/sb_active.mp4" type="video/mp4">
    </video>
</div>
and the corresponding **active** training of a DNN on the MNIST dataset:
<div style="text-align: center;">
    <video width="640" height="480" controls>
        <source src="videos/DNN_active.mp4" type="video/mp4">
    </video>
</div>

## Reproducing phase diagrams in real neural network
![Sample Data Visualization](figs/pd.png)
```
./bash/test_pd_ns_nl.sh   # data noise
./bash/test_pd_lr_nl.sh   # learning rate
./bash/test_pd_dp_nl.sh   # drop out
./bash/test_pd_bm_nl.sh   # batch size
```

## Spring block simulation
In the [first notebook](./spring/first_order_iteration.ipynb),  we solve the first-order overdamping system by discrete iteration.

In the [second notebook](./spring/sdeint.ipynb), we solve the second-order stochastic differential equation using [sdeint](https://github.com/mattja/sdeint).