# GPU PDE Lab

This is a more or less vibecoded platform for GPU-accelerated PDE visualization. It is inspired by the course "solving PDEs on GPUs in parallel" of ETH Zurich.

When I first heard of this course, I though that it involves finding approximate PDE solutions of particular classes by some clever parallelization techniques, but it turns out to primarily describe how to solve the starting states of a medium numerically, given some PDEs. Since this involves solving them for every particle in the field, this gives huge opportunities for parallelization.

The content of the app includes mainly visualization of the most used PDEs in CS and Physics, and an editor that allows you to define or import your own desired start state in 2D. 3D is work in progress and partially implemented (probably most interesting for heat equation modelling of 3D objects). External forces or flows are yet to be implemented. This Platform is intended primarily for GPU accelerated compute, although CPU fallbacks exist, they might be kind of slow and are not intended for practical usage or performant visualization.

Right now, the setup guide only supports NixOS, since this is the OS used in developement. The app is a python app. Feel free to adapt the environment needed, might add support for other environments later. Setup guide and further informations (AI generated) can be found under `./doc.md`. 
