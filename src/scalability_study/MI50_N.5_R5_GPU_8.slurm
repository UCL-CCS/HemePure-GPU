

======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK    MCLK     Fan   Perf  PwrCap  VRAM%  GPU%  
0    36.0c  35.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
1    37.0c  35.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
2    37.0c  34.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
3    39.0c  40.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
4    37.0c  34.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
5    37.0c  34.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
6    37.0c  38.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
7    36.0c  34.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
================================================================================
============================= End of ROCm SMI Log ==============================
[ts1-sjc2-16:28621] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[ts1-sjc2-16:28621] MCW rank 1 bound to socket 0[core 16[hwt 0]]: [././././././././././././././././B/././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[ts1-sjc2-16:28621] MCW rank 2 bound to socket 0[core 32[hwt 0]]: [././././././././././././././././././././././././././././././././B/././././././././././././././././././././././././././././././.][./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[ts1-sjc2-16:28621] MCW rank 3 bound to socket 0[core 48[hwt 0]]: [././././././././././././././././././././././././././././././././././././././././././././././././B/././././././././././././././.][./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[ts1-sjc2-16:28621] MCW rank 4 bound to socket 1[core 64[hwt 0]]: [./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][B/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[Rank 0000000, 3.602762e-03 s, 00000002236740 kB] :: INITIALISE
[Rank 0000000, 3.647908e-03 s, 00000002236740 kB] :: ----------
[Rank 0000000, 3.661804e-03 s, 00000002236740 kB] :: --> loading input and decomposing geometry
[Rank 0000000, 1.799127e-02 s, 00000002244064 kB] :: ----> opened data file ../../cases/FileSharingIZPK/CBM2019_Arteries_patched.gmy
[Rank 0000000, 1.803429e-02 s, 00000002244064 kB] :: ----> reading preamble
[Rank 0000000, 1.819045e-02 s, 00000002244064 kB] :: ----> reading header (start)
[Rank 0000000, 4.004219e+00 s, 00000002309356 kB] :: ----> reading header (end)
[Rank 0000000, 5.151100e+00 s, 00000002309352 kB] :: ----> non-empty blocks: 199393
[Rank 0000000, 5.151134e+00 s, 00000002309352 kB] ::           total blocks: 119617344
[Rank 0000000, 5.151147e+00 s, 00000002309352 kB] ::                  ratio: 0.001667
[Rank 0000000, 5.151161e+00 s, 00000002309352 kB] ::                  sites: 66401494
[Rank 0000000, 5.151173e+00 s, 00000002309352 kB] :: ----> blockInformation.size(): 199393
[Rank 0000000, 5.151187e+00 s, 00000002309352 kB] ::  fluidSitesOnEachBlock.size(): 0
[Rank 0000000, 5.151199e+00 s, 00000002309352 kB] ::           blockWeights.size(): 0
[Rank 0000000, 5.151211e+00 s, 00000002309352 kB] :: ----> is blockInformation.size() == nonEmptyBlocks? yes
[Rank 0000000, 5.151223e+00 s, 00000002309352 kB] :: ----> not optimising decomposition
[Rank 0000000, 5.153944e+00 s, 00000002309352 kB] :: ----> basic decomposition (start)
[Rank 0000001, 8.602012e+00 s, 00000000479264 kB] :: ----> load distribution: 0.000010
[Rank 0000000, 8.830496e+00 s, 00000002374608 kB] :: ----> basic decomposition (end)
[Rank 0000000, 8.830529e+00 s, 00000002374608 kB] :: ----> read blocks (start)
[Rank 0000001, 1.400692e+01 s, 00000000491548 kB] :: ----> blockInformation.size(): 199393
[Rank 0000002, 1.400695e+01 s, 00000000491748 kB] :: ----> blockInformation.size(): 199393
[Rank 0000003, 1.400695e+01 s, 00000000479388 kB] :: ----> blockInformation.size(): 50813
[Rank 0000004, 1.400693e+01 s, 00000000479536 kB] :: ----> blockInformation.size(): 50540
[Rank 0000000, 7.570348e+01 s, 00000022268060 kB] :: ----> read blocks (end)
[Rank 0000000, 8.112878e+01 s, 00000022257504 kB] :: --> lattice data
[Rank 0000000, 1.018714e+02 s, 00000058484580 kB] :: ----> gathering lattice information (start)
[Rank 0000000, 1.018715e+02 s, 00000058484580 kB] :: ----> gathering lattice information (end)
[Rank 0000000, 1.303871e+02 s, 00000068344436 kB] :: --> neighbouring data manager
[Rank 0000000, 1.303871e+02 s, 00000068344436 kB] :: --> lattice-Boltzmann model
Rank 0: Detected 8 CUDA Capable device(s)
Rank: 4, Iolet ID: 18, Frequency: 2 
Rank: 4, Iolet ID: 19, Frequency: 2 
Rank: 4, Iolet ID: 23, Frequency: 2 
Rank: 4, Iolet ID: 24, Frequency: 1 
Rank: 4, Iolet ID: 25, Frequency: 2 
Rank: 4, Iolet ID: 26, Frequency: 2 
Rank: 4, Iolet ID: 27, Frequency: 1 
Rank: 4, Iolet ID: 28, Frequency: 1 
Rank: 4, Iolet ID: 29, Frequency: 1 
Rank: 4, Iolet ID: 32, Frequency: 1 
Rank: 4, Iolet ID: 33, Frequency: 1 
Rank: 4, Iolet ID: 34, Frequency: 1 
Rank: 4, Iolet ID: 35, Frequency: 1 
Rank: 4, Iolet ID: 36, Frequency: 1 
Rank: 4, Iolet ID: 37, Frequency: 1 
Rank: 4, Iolet ID: 18, Frequency: 3 
Rank: 4, Iolet ID: 19, Frequency: 2 
Rank: 4, Iolet ID: 23, Frequency: 3 
Rank: 4, Iolet ID: 24, Frequency: 2 
Rank: 4, Iolet ID: 25, Frequency: 2 
Rank: 4, Iolet ID: 26, Frequency: 2 
Rank: 4, Iolet ID: 27, Frequency: 1 
Rank: 4, Iolet ID: 28, Frequency: 1 
Rank: 4, Iolet ID: 29, Frequency: 1 
Rank: 4, Iolet ID: 32, Frequency: 1 
Rank: 4, Iolet ID: 33, Frequency: 1 
Rank: 4, Iolet ID: 34, Frequency: 1 
Rank: 4, Iolet ID: 35, Frequency: 1 
Rank: 4, Iolet ID: 36, Frequency: 1 
Rank: 4, Iolet ID: 37, Frequency: 1 
INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 4 - Total requested global memory 7.29GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 2 - Total requested global memory 7.70GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 1 - Total requested global memory 7.74GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 3 - Total requested global memory 7.97GB 

[Rank 0000000, 1.445084e+02 s, 00000127860908 kB] :: -------------------
[Rank 0000000, 1.445084e+02 s, 00000127860908 kB] :: INITIALISE FINISHED
[Rank 0000000, 1.450539e+02 s, 00000127845224 kB] :: SIMULATION STARTING
[Rank 0000000, 1.450539e+02 s, 00000127845224 kB] :: -------------------
[Rank 0000000, 1.568496e+02 s, 00000127975084 kB] :: time step 0000200 :: write_image_to_disk 0
[Rank 0000000, 1.685012e+02 s, 00000127975776 kB] :: time step 0000400 :: write_image_to_disk 0
[Rank 0000000, 1.801457e+02 s, 00000127976136 kB] :: time step 0000600 :: write_image_to_disk 0
[Rank 0000000, 1.918003e+02 s, 00000127976136 kB] :: time step 0000800 :: write_image_to_disk 0
[Rank 0000000, 2.034648e+02 s, 00000127976136 kB] :: time step 0001000 :: write_image_to_disk 0
[Rank 0000000, 2.153329e+02 s, 00000129014060 kB] :: time step 0001200 :: write_image_to_disk 0
[Rank 0000000, 2.269845e+02 s, 00000129014060 kB] :: time step 0001400 :: write_image_to_disk 0
[Rank 0000000, 2.386292e+02 s, 00000129014060 kB] :: time step 0001600 :: write_image_to_disk 0
[Rank 0000000, 2.502866e+02 s, 00000129014060 kB] :: time step 0001800 :: write_image_to_disk 0
[Rank 0000000, 2.619406e+02 s, 00000129014060 kB] :: time step 0002000 :: write_image_to_disk 0
[Rank 0000000, 2.738070e+02 s, 00000130051600 kB] :: time step 0002200 :: write_image_to_disk 0
[Rank 0000000, 2.854604e+02 s, 00000130051600 kB] :: time step 0002400 :: write_image_to_disk 0
[Rank 0000000, 2.971103e+02 s, 00000130051600 kB] :: time step 0002600 :: write_image_to_disk 0
[Rank 0000000, 3.087659e+02 s, 00000130051600 kB] :: time step 0002800 :: write_image_to_disk 0
[Rank 0000000, 3.204306e+02 s, 00000130051600 kB] :: time step 0003000 :: write_image_to_disk 0
[Rank 0000000, 3.323148e+02 s, 00000131089352 kB] :: time step 0003200 :: write_image_to_disk 0
[Rank 0000000, 3.439706e+02 s, 00000131089352 kB] :: time step 0003400 :: write_image_to_disk 0
[Rank 0000000, 3.556288e+02 s, 00000131089352 kB] :: time step 0003600 :: write_image_to_disk 0
[Rank 0000000, 3.672844e+02 s, 00000131089352 kB] :: time step 0003800 :: write_image_to_disk 0
[Rank 0000000, 3.789353e+02 s, 00000131089352 kB] :: time step 0004000 :: write_image_to_disk 0
[Rank 0000000, 3.908036e+02 s, 00000132126888 kB] :: time step 0004200 :: write_image_to_disk 0
[Rank 0000000, 4.024597e+02 s, 00000132126888 kB] :: time step 0004400 :: write_image_to_disk 0
[Rank 0000000, 4.141134e+02 s, 00000132126888 kB] :: time step 0004600 :: write_image_to_disk 0
[Rank 0000000, 4.257742e+02 s, 00000132126888 kB] :: time step 0004800 :: write_image_to_disk 0
[Rank 0000000, 4.374377e+02 s, 00000132126888 kB] :: time step 0005000 :: write_image_to_disk 0
[Rank 0000000, 4.493145e+02 s, 00000133164424 kB] :: time step 0005200 :: write_image_to_disk 0
[Rank 0000000, 4.609627e+02 s, 00000133164424 kB] :: time step 0005400 :: write_image_to_disk 0
[Rank 0000000, 4.726194e+02 s, 00000133164424 kB] :: time step 0005600 :: write_image_to_disk 0
[Rank 0000000, 4.842817e+02 s, 00000133164424 kB] :: time step 0005800 :: write_image_to_disk 0
[Rank 0000000, 4.959379e+02 s, 00000133164424 kB] :: time step 0006000 :: write_image_to_disk 0
[Rank 0000000, 5.078041e+02 s, 00000134201960 kB] :: time step 0006200 :: write_image_to_disk 0
[Rank 0000000, 5.194565e+02 s, 00000134201960 kB] :: time step 0006400 :: write_image_to_disk 0
[Rank 0000000, 5.311102e+02 s, 00000134201960 kB] :: time step 0006600 :: write_image_to_disk 0
[Rank 0000000, 5.427739e+02 s, 00000134201960 kB] :: time step 0006800 :: write_image_to_disk 0
[Rank 0000000, 5.544273e+02 s, 00000134201960 kB] :: time step 0007000 :: write_image_to_disk 0
[Rank 0000000, 5.662959e+02 s, 00000135239496 kB] :: time step 0007200 :: write_image_to_disk 0
[Rank 0000000, 5.779474e+02 s, 00000135239496 kB] :: time step 0007400 :: write_image_to_disk 0
[Rank 0000000, 5.896025e+02 s, 00000135239496 kB] :: time step 0007600 :: write_image_to_disk 0
[Rank 0000000, 6.012534e+02 s, 00000135239496 kB] :: time step 0007800 :: write_image_to_disk 0
[Rank 0000000, 6.129096e+02 s, 00000135239496 kB] :: time step 0008000 :: write_image_to_disk 0
[Rank 0000000, 6.247785e+02 s, 00000136277032 kB] :: time step 0008200 :: write_image_to_disk 0
[Rank 0000000, 6.364323e+02 s, 00000136277032 kB] :: time step 0008400 :: write_image_to_disk 0
[Rank 0000000, 6.480914e+02 s, 00000136277032 kB] :: time step 0008600 :: write_image_to_disk 0
[Rank 0000000, 6.597460e+02 s, 00000136277032 kB] :: time step 0008800 :: write_image_to_disk 0
[Rank 0000000, 6.714045e+02 s, 00000136277032 kB] :: time step 0009000 :: write_image_to_disk 0
[Rank 0000000, 6.832731e+02 s, 00000137314568 kB] :: time step 0009200 :: write_image_to_disk 0
[Rank 0000000, 6.949246e+02 s, 00000137314568 kB] :: time step 0009400 :: write_image_to_disk 0
[Rank 0000000, 7.065786e+02 s, 00000137314568 kB] :: time step 0009600 :: write_image_to_disk 0
[Rank 0000000, 7.182366e+02 s, 00000137314568 kB] :: time step 0009800 :: write_image_to_disk 0
[Rank 0000000, 7.299011e+02 s, 00000137314568 kB] :: time step 0010000 :: write_image_to_disk 1
[Rank 0000000, 7.418108e+02 s, 00000138352104 kB] :: time step 0010200 :: write_image_to_disk 0
[Rank 0000000, 7.534713e+02 s, 00000138352104 kB] :: time step 0010400 :: write_image_to_disk 0
[Rank 0000000, 7.651343e+02 s, 00000138352104 kB] :: time step 0010600 :: write_image_to_disk 0
[Rank 0000000, 7.767943e+02 s, 00000138352104 kB] :: time step 0010800 :: write_image_to_disk 0
[Rank 0000000, 7.884470e+02 s, 00000138352104 kB] :: time step 0011000 :: write_image_to_disk 0
[Rank 0000000, 8.003526e+02 s, 00000139389640 kB] :: time step 0011200 :: write_image_to_disk 0
[Rank 0000000, 8.120070e+02 s, 00000139389640 kB] :: time step 0011400 :: write_image_to_disk 0
[Rank 0000000, 8.236675e+02 s, 00000139389640 kB] :: time step 0011600 :: write_image_to_disk 0
[Rank 0000000, 8.353272e+02 s, 00000139389640 kB] :: time step 0011800 :: write_image_to_disk 0
[Rank 0000000, 8.469893e+02 s, 00000139389640 kB] :: time step 0012000 :: write_image_to_disk 0
[Rank 0000000, 8.589001e+02 s, 00000140427176 kB] :: time step 0012200 :: write_image_to_disk 0
[Rank 0000000, 8.705661e+02 s, 00000140427176 kB] :: time step 0012400 :: write_image_to_disk 0
[Rank 0000000, 8.822307e+02 s, 00000140427176 kB] :: time step 0012600 :: write_image_to_disk 0
[Rank 0000000, 8.938974e+02 s, 00000140427176 kB] :: time step 0012800 :: write_image_to_disk 0
[Rank 0000000, 9.055516e+02 s, 00000140427176 kB] :: time step 0013000 :: write_image_to_disk 0
[Rank 0000000, 9.174623e+02 s, 00000141464712 kB] :: time step 0013200 :: write_image_to_disk 0
[Rank 0000000, 9.291226e+02 s, 00000141464712 kB] :: time step 0013400 :: write_image_to_disk 0
[Rank 0000000, 9.407823e+02 s, 00000141464712 kB] :: time step 0013600 :: write_image_to_disk 0
[Rank 0000000, 9.524356e+02 s, 00000141464712 kB] :: time step 0013800 :: write_image_to_disk 0
[Rank 0000000, 9.641021e+02 s, 00000141464712 kB] :: time step 0014000 :: write_image_to_disk 0
[Rank 0000000, 9.760095e+02 s, 00000142502208 kB] :: time step 0014200 :: write_image_to_disk 0
[Rank 0000000, 9.876721e+02 s, 00000142502208 kB] :: time step 0014400 :: write_image_to_disk 0
[Rank 0000000, 9.993343e+02 s, 00000142502208 kB] :: time step 0014600 :: write_image_to_disk 0
[Rank 0000000, 1.010997e+03 s, 00000142502208 kB] :: time step 0014800 :: write_image_to_disk 0
[Rank 0000000, 1.022660e+03 s, 00000142502208 kB] :: time step 0015000 :: write_image_to_disk 0
[Rank 0000000, 1.034562e+03 s, 00000143539744 kB] :: time step 0015200 :: write_image_to_disk 0
[Rank 0000000, 1.046218e+03 s, 00000143539744 kB] :: time step 0015400 :: write_image_to_disk 0
[Rank 0000000, 1.057876e+03 s, 00000143539744 kB] :: time step 0015600 :: write_image_to_disk 0
[Rank 0000000, 1.069538e+03 s, 00000143539744 kB] :: time step 0015800 :: write_image_to_disk 0
[Rank 0000000, 1.081196e+03 s, 00000143539744 kB] :: time step 0016000 :: write_image_to_disk 0
[Rank 0000000, 1.093088e+03 s, 00000144577280 kB] :: time step 0016200 :: write_image_to_disk 0
[Rank 0000000, 1.104747e+03 s, 00000144577280 kB] :: time step 0016400 :: write_image_to_disk 0
[Rank 0000000, 1.116408e+03 s, 00000144577280 kB] :: time step 0016600 :: write_image_to_disk 0
[Rank 0000000, 1.128071e+03 s, 00000144577280 kB] :: time step 0016800 :: write_image_to_disk 0
[Rank 0000000, 1.139725e+03 s, 00000144577280 kB] :: time step 0017000 :: write_image_to_disk 0
[Rank 0000000, 1.151626e+03 s, 00000145614816 kB] :: time step 0017200 :: write_image_to_disk 0
[Rank 0000000, 1.163284e+03 s, 00000145614816 kB] :: time step 0017400 :: write_image_to_disk 0
[Rank 0000000, 1.174942e+03 s, 00000145614816 kB] :: time step 0017600 :: write_image_to_disk 0
[Rank 0000000, 1.186591e+03 s, 00000145614816 kB] :: time step 0017800 :: write_image_to_disk 0
[Rank 0000000, 1.198249e+03 s, 00000145614816 kB] :: time step 0018000 :: write_image_to_disk 0
[Rank 0000000, 1.210140e+03 s, 00000146652352 kB] :: time step 0018200 :: write_image_to_disk 0
[Rank 0000000, 1.221797e+03 s, 00000146652352 kB] :: time step 0018400 :: write_image_to_disk 0
[Rank 0000000, 1.233453e+03 s, 00000146652352 kB] :: time step 0018600 :: write_image_to_disk 0
[Rank 0000000, 1.245107e+03 s, 00000146652352 kB] :: time step 0018800 :: write_image_to_disk 0
[Rank 0000000, 1.256761e+03 s, 00000146652352 kB] :: time step 0019000 :: write_image_to_disk 0
[Rank 0000000, 1.268659e+03 s, 00000147689888 kB] :: time step 0019200 :: write_image_to_disk 0
[Rank 0000000, 1.280319e+03 s, 00000147689888 kB] :: time step 0019400 :: write_image_to_disk 0
[Rank 0000000, 1.291982e+03 s, 00000147689888 kB] :: time step 0019600 :: write_image_to_disk 0
[Rank 0000000, 1.303641e+03 s, 00000147689888 kB] :: time step 0019800 :: write_image_to_disk 0
[Rank 0000000, 1.315292e+03 s, 00000147689888 kB] :: time step 0020000 :: write_image_to_disk 1
[Rank 0000000, 1.327194e+03 s, 00000148727424 kB] :: time step 0020200 :: write_image_to_disk 0
[Rank 0000000, 1.338853e+03 s, 00000148727424 kB] :: time step 0020400 :: write_image_to_disk 0
[Rank 0000000, 1.350517e+03 s, 00000148727424 kB] :: time step 0020600 :: write_image_to_disk 0
[Rank 0000000, 1.362176e+03 s, 00000148727424 kB] :: time step 0020800 :: write_image_to_disk 0
[Rank 0000000, 1.373840e+03 s, 00000148727424 kB] :: time step 0021000 :: write_image_to_disk 0
[Rank 0000000, 1.385737e+03 s, 00000149764960 kB] :: time step 0021200 :: write_image_to_disk 0
[Rank 0000000, 1.397400e+03 s, 00000149764960 kB] :: time step 0021400 :: write_image_to_disk 0
[Rank 0000000, 1.409061e+03 s, 00000149764960 kB] :: time step 0021600 :: write_image_to_disk 0
[Rank 0000000, 1.420717e+03 s, 00000149764960 kB] :: time step 0021800 :: write_image_to_disk 0
[Rank 0000000, 1.432377e+03 s, 00000149764960 kB] :: time step 0022000 :: write_image_to_disk 0
[Rank 0000000, 1.444282e+03 s, 00000150802496 kB] :: time step 0022200 :: write_image_to_disk 0
[Rank 0000000, 1.455943e+03 s, 00000150802496 kB] :: time step 0022400 :: write_image_to_disk 0
[Rank 0000000, 1.467604e+03 s, 00000150802496 kB] :: time step 0022600 :: write_image_to_disk 0
[Rank 0000000, 1.479260e+03 s, 00000150802496 kB] :: time step 0022800 :: write_image_to_disk 0
[Rank 0000000, 1.490926e+03 s, 00000150802496 kB] :: time step 0023000 :: write_image_to_disk 0
[Rank 0000000, 1.502824e+03 s, 00000151840032 kB] :: time step 0023200 :: write_image_to_disk 0
[Rank 0000000, 1.514485e+03 s, 00000151840032 kB] :: time step 0023400 :: write_image_to_disk 0
[Rank 0000000, 1.526139e+03 s, 00000151840032 kB] :: time step 0023600 :: write_image_to_disk 0
[Rank 0000000, 1.537804e+03 s, 00000151840032 kB] :: time step 0023800 :: write_image_to_disk 0
[Rank 0000000, 1.549462e+03 s, 00000151840032 kB] :: time step 0024000 :: write_image_to_disk 0
[Rank 0000000, 1.561363e+03 s, 00000152877568 kB] :: time step 0024200 :: write_image_to_disk 0
[Rank 0000000, 1.573023e+03 s, 00000152877568 kB] :: time step 0024400 :: write_image_to_disk 0
[Rank 0000000, 1.584683e+03 s, 00000152877568 kB] :: time step 0024600 :: write_image_to_disk 0
[Rank 0000000, 1.596347e+03 s, 00000152877568 kB] :: time step 0024800 :: write_image_to_disk 0
[Rank 0000000, 1.608003e+03 s, 00000152877568 kB] :: time step 0025000 :: write_image_to_disk 0
[Rank 0000000, 1.619907e+03 s, 00000153915104 kB] :: time step 0025200 :: write_image_to_disk 0
[Rank 0000000, 1.631566e+03 s, 00000153915104 kB] :: time step 0025400 :: write_image_to_disk 0
[Rank 0000000, 1.643223e+03 s, 00000153915104 kB] :: time step 0025600 :: write_image_to_disk 0
[Rank 0000000, 1.654879e+03 s, 00000153915104 kB] :: time step 0025800 :: write_image_to_disk 0
[Rank 0000000, 1.666531e+03 s, 00000153915104 kB] :: time step 0026000 :: write_image_to_disk 0
[Rank 0000000, 1.678434e+03 s, 00000154952640 kB] :: time step 0026200 :: write_image_to_disk 0
[Rank 0000000, 1.690089e+03 s, 00000154952640 kB] :: time step 0026400 :: write_image_to_disk 0
[Rank 0000000, 1.701749e+03 s, 00000154952640 kB] :: time step 0026600 :: write_image_to_disk 0
[Rank 0000000, 1.713404e+03 s, 00000154952640 kB] :: time step 0026800 :: write_image_to_disk 0
[Rank 0000000, 1.725063e+03 s, 00000154952640 kB] :: time step 0027000 :: write_image_to_disk 0
[Rank 0000000, 1.736960e+03 s, 00000155990176 kB] :: time step 0027200 :: write_image_to_disk 0
[Rank 0000000, 1.748613e+03 s, 00000155990176 kB] :: time step 0027400 :: write_image_to_disk 0
[Rank 0000000, 1.760268e+03 s, 00000155990176 kB] :: time step 0027600 :: write_image_to_disk 0
[Rank 0000000, 1.771925e+03 s, 00000155990176 kB] :: time step 0027800 :: write_image_to_disk 0
[Rank 0000000, 1.783584e+03 s, 00000155990176 kB] :: time step 0028000 :: write_image_to_disk 0
[Rank 0000000, 1.795494e+03 s, 00000157027712 kB] :: time step 0028200 :: write_image_to_disk 0
[Rank 0000000, 1.807144e+03 s, 00000157027712 kB] :: time step 0028400 :: write_image_to_disk 0
[Rank 0000000, 1.818808e+03 s, 00000157027712 kB] :: time step 0028600 :: write_image_to_disk 0
[Rank 0000000, 1.830468e+03 s, 00000157027712 kB] :: time step 0028800 :: write_image_to_disk 0
[Rank 0000000, 1.842129e+03 s, 00000157027712 kB] :: time step 0029000 :: write_image_to_disk 0
[Rank 0000000, 1.854030e+03 s, 00000158065248 kB] :: time step 0029200 :: write_image_to_disk 0
[Rank 0000000, 1.865703e+03 s, 00000158065248 kB] :: time step 0029400 :: write_image_to_disk 0
[Rank 0000000, 1.877354e+03 s, 00000158065248 kB] :: time step 0029600 :: write_image_to_disk 0
[Rank 0000000, 1.889010e+03 s, 00000158065248 kB] :: time step 0029800 :: write_image_to_disk 0
[Rank 0000000, 1.900671e+03 s, 00000158065248 kB] :: time step 0030000 :: write_image_to_disk 1
[Rank 0000000, 1.912568e+03 s, 00000159102784 kB] :: time step 0030200 :: write_image_to_disk 0
[Rank 0000000, 1.924227e+03 s, 00000159102784 kB] :: time step 0030400 :: write_image_to_disk 0
[Rank 0000000, 1.935885e+03 s, 00000159102784 kB] :: time step 0030600 :: write_image_to_disk 0
[Rank 0000000, 1.947544e+03 s, 00000159102784 kB] :: time step 0030800 :: write_image_to_disk 0
[Rank 0000000, 1.959202e+03 s, 00000159102784 kB] :: time step 0031000 :: write_image_to_disk 0
[Rank 0000000, 1.971097e+03 s, 00000160140320 kB] :: time step 0031200 :: write_image_to_disk 0
[Rank 0000000, 1.982757e+03 s, 00000160140320 kB] :: time step 0031400 :: write_image_to_disk 0
[Rank 0000000, 1.994410e+03 s, 00000160140320 kB] :: time step 0031600 :: write_image_to_disk 0
[Rank 0000000, 2.006074e+03 s, 00000160140320 kB] :: time step 0031800 :: write_image_to_disk 0
[Rank 0000000, 2.017741e+03 s, 00000160140320 kB] :: time step 0032000 :: write_image_to_disk 0
[Rank 0000000, 2.029638e+03 s, 00000161177856 kB] :: time step 0032200 :: write_image_to_disk 0
[Rank 0000000, 2.041298e+03 s, 00000161177856 kB] :: time step 0032400 :: write_image_to_disk 0
[Rank 0000000, 2.052955e+03 s, 00000161177856 kB] :: time step 0032600 :: write_image_to_disk 0
[Rank 0000000, 2.064615e+03 s, 00000161177856 kB] :: time step 0032800 :: write_image_to_disk 0
[Rank 0000000, 2.076280e+03 s, 00000161177856 kB] :: time step 0033000 :: write_image_to_disk 0
[Rank 0000000, 2.088167e+03 s, 00000162215392 kB] :: time step 0033200 :: write_image_to_disk 0
[Rank 0000000, 2.099830e+03 s, 00000162215392 kB] :: time step 0033400 :: write_image_to_disk 0
[Rank 0000000, 2.111493e+03 s, 00000162215392 kB] :: time step 0033600 :: write_image_to_disk 0
[Rank 0000000, 2.123149e+03 s, 00000162215392 kB] :: time step 0033800 :: write_image_to_disk 0
[Rank 0000000, 2.134812e+03 s, 00000162215392 kB] :: time step 0034000 :: write_image_to_disk 0
[Rank 0000000, 2.146699e+03 s, 00000163252928 kB] :: time step 0034200 :: write_image_to_disk 0
[Rank 0000000, 2.158363e+03 s, 00000163252928 kB] :: time step 0034400 :: write_image_to_disk 0
[Rank 0000000, 2.170022e+03 s, 00000163252928 kB] :: time step 0034600 :: write_image_to_disk 0
[Rank 0000000, 2.181687e+03 s, 00000163252928 kB] :: time step 0034800 :: write_image_to_disk 0
[Rank 0000000, 2.193345e+03 s, 00000163252928 kB] :: time step 0035000 :: write_image_to_disk 0
[Rank 0000000, 2.205235e+03 s, 00000164290464 kB] :: time step 0035200 :: write_image_to_disk 0
[Rank 0000000, 2.216894e+03 s, 00000164290464 kB] :: time step 0035400 :: write_image_to_disk 0
[Rank 0000000, 2.228547e+03 s, 00000164290464 kB] :: time step 0035600 :: write_image_to_disk 0
[Rank 0000000, 2.240205e+03 s, 00000164290464 kB] :: time step 0035800 :: write_image_to_disk 0
[Rank 0000000, 2.251864e+03 s, 00000164290464 kB] :: time step 0036000 :: write_image_to_disk 0
[Rank 0000000, 2.263755e+03 s, 00000165328000 kB] :: time step 0036200 :: write_image_to_disk 0
[Rank 0000000, 2.275417e+03 s, 00000165328000 kB] :: time step 0036400 :: write_image_to_disk 0
[Rank 0000000, 2.287079e+03 s, 00000165328000 kB] :: time step 0036600 :: write_image_to_disk 0
[Rank 0000000, 2.298734e+03 s, 00000165328000 kB] :: time step 0036800 :: write_image_to_disk 0
[Rank 0000000, 2.310386e+03 s, 00000165328000 kB] :: time step 0037000 :: write_image_to_disk 0
[Rank 0000000, 2.322278e+03 s, 00000166365536 kB] :: time step 0037200 :: write_image_to_disk 0
[Rank 0000000, 2.333945e+03 s, 00000166365536 kB] :: time step 0037400 :: write_image_to_disk 0
[Rank 0000000, 2.345605e+03 s, 00000166365536 kB] :: time step 0037600 :: write_image_to_disk 0
[Rank 0000000, 2.357269e+03 s, 00000166365536 kB] :: time step 0037800 :: write_image_to_disk 0
[Rank 0000000, 2.368920e+03 s, 00000166365536 kB] :: time step 0038000 :: write_image_to_disk 0
[Rank 0000000, 2.380801e+03 s, 00000167403072 kB] :: time step 0038200 :: write_image_to_disk 0
[Rank 0000000, 2.392467e+03 s, 00000167403072 kB] :: time step 0038400 :: write_image_to_disk 0
[Rank 0000000, 2.404124e+03 s, 00000167403072 kB] :: time step 0038600 :: write_image_to_disk 0
[Rank 0000000, 2.415783e+03 s, 00000167403072 kB] :: time step 0038800 :: write_image_to_disk 0
[Rank 0000000, 2.427438e+03 s, 00000167403072 kB] :: time step 0039000 :: write_image_to_disk 0
[Rank 0000000, 2.439316e+03 s, 00000168440608 kB] :: time step 0039200 :: write_image_to_disk 0
[Rank 0000000, 2.450978e+03 s, 00000168440608 kB] :: time step 0039400 :: write_image_to_disk 0
[Rank 0000000, 2.462640e+03 s, 00000168440608 kB] :: time step 0039600 :: write_image_to_disk 0
[Rank 0000000, 2.474300e+03 s, 00000168440608 kB] :: time step 0039800 :: write_image_to_disk 0
[Rank 0000000, 2.485963e+03 s, 00000168440608 kB] :: time step 0040000 :: write_image_to_disk 1
[Rank 0000000, 2.497837e+03 s, 00000169478144 kB] :: time step 0040200 :: write_image_to_disk 0
[Rank 0000000, 2.509500e+03 s, 00000169478144 kB] :: time step 0040400 :: write_image_to_disk 0
[Rank 0000000, 2.521157e+03 s, 00000169478144 kB] :: time step 0040600 :: write_image_to_disk 0
[Rank 0000000, 2.532811e+03 s, 00000169478144 kB] :: time step 0040800 :: write_image_to_disk 0
[Rank 0000000, 2.544470e+03 s, 00000169478144 kB] :: time step 0041000 :: write_image_to_disk 0
[Rank 0000000, 2.556353e+03 s, 00000170515680 kB] :: time step 0041200 :: write_image_to_disk 0
[Rank 0000000, 2.568010e+03 s, 00000170515680 kB] :: time step 0041400 :: write_image_to_disk 0
[Rank 0000000, 2.579677e+03 s, 00000170515680 kB] :: time step 0041600 :: write_image_to_disk 0
[Rank 0000000, 2.591341e+03 s, 00000170515680 kB] :: time step 0041800 :: write_image_to_disk 0
[Rank 0000000, 2.603000e+03 s, 00000170515680 kB] :: time step 0042000 :: write_image_to_disk 0
[Rank 0000000, 2.614877e+03 s, 00000171553216 kB] :: time step 0042200 :: write_image_to_disk 0
[Rank 0000000, 2.626533e+03 s, 00000171553216 kB] :: time step 0042400 :: write_image_to_disk 0
[Rank 0000000, 2.638179e+03 s, 00000171553216 kB] :: time step 0042600 :: write_image_to_disk 0
[Rank 0000000, 2.649847e+03 s, 00000171553216 kB] :: time step 0042800 :: write_image_to_disk 0
[Rank 0000000, 2.661502e+03 s, 00000171553216 kB] :: time step 0043000 :: write_image_to_disk 0
[Rank 0000000, 2.673391e+03 s, 00000172590752 kB] :: time step 0043200 :: write_image_to_disk 0
[Rank 0000000, 2.685048e+03 s, 00000172590752 kB] :: time step 0043400 :: write_image_to_disk 0
[Rank 0000000, 2.696708e+03 s, 00000172590752 kB] :: time step 0043600 :: write_image_to_disk 0
[Rank 0000000, 2.708373e+03 s, 00000172590752 kB] :: time step 0043800 :: write_image_to_disk 0
[Rank 0000000, 2.720031e+03 s, 00000172590752 kB] :: time step 0044000 :: write_image_to_disk 0
[Rank 0000000, 2.731909e+03 s, 00000173628288 kB] :: time step 0044200 :: write_image_to_disk 0
[Rank 0000000, 2.743575e+03 s, 00000173628288 kB] :: time step 0044400 :: write_image_to_disk 0
[Rank 0000000, 2.755240e+03 s, 00000173628288 kB] :: time step 0044600 :: write_image_to_disk 0
[Rank 0000000, 2.766902e+03 s, 00000173628288 kB] :: time step 0044800 :: write_image_to_disk 0
[Rank 0000000, 2.778551e+03 s, 00000173628288 kB] :: time step 0045000 :: write_image_to_disk 0
[Rank 0000000, 2.790462e+03 s, 00000174665824 kB] :: time step 0045200 :: write_image_to_disk 0
[Rank 0000000, 2.802117e+03 s, 00000174665824 kB] :: time step 0045400 :: write_image_to_disk 0
[Rank 0000000, 2.813775e+03 s, 00000174665824 kB] :: time step 0045600 :: write_image_to_disk 0
[Rank 0000000, 2.825425e+03 s, 00000174665824 kB] :: time step 0045800 :: write_image_to_disk 0
[Rank 0000000, 2.837088e+03 s, 00000174665824 kB] :: time step 0046000 :: write_image_to_disk 0
[Rank 0000000, 2.848998e+03 s, 00000175703360 kB] :: time step 0046200 :: write_image_to_disk 0
[Rank 0000000, 2.860650e+03 s, 00000175703360 kB] :: time step 0046400 :: write_image_to_disk 0
[Rank 0000000, 2.872313e+03 s, 00000175703360 kB] :: time step 0046600 :: write_image_to_disk 0
[Rank 0000000, 2.883974e+03 s, 00000175703360 kB] :: time step 0046800 :: write_image_to_disk 0
[Rank 0000000, 2.895637e+03 s, 00000175703360 kB] :: time step 0047000 :: write_image_to_disk 0
[Rank 0000000, 2.907545e+03 s, 00000176740896 kB] :: time step 0047200 :: write_image_to_disk 0
[Rank 0000000, 2.919202e+03 s, 00000176740896 kB] :: time step 0047400 :: write_image_to_disk 0
[Rank 0000000, 2.930869e+03 s, 00000176740896 kB] :: time step 0047600 :: write_image_to_disk 0
[Rank 0000000, 2.942529e+03 s, 00000176740896 kB] :: time step 0047800 :: write_image_to_disk 0
[Rank 0000000, 2.954192e+03 s, 00000176740896 kB] :: time step 0048000 :: write_image_to_disk 0
[Rank 0000000, 2.966101e+03 s, 00000177778432 kB] :: time step 0048200 :: write_image_to_disk 0
[Rank 0000000, 2.977763e+03 s, 00000177778432 kB] :: time step 0048400 :: write_image_to_disk 0
[Rank 0000000, 2.989427e+03 s, 00000177778432 kB] :: time step 0048600 :: write_image_to_disk 0
[Rank 0000000, 3.001089e+03 s, 00000177778432 kB] :: time step 0048800 :: write_image_to_disk 0
[Rank 0000000, 3.012748e+03 s, 00000177778432 kB] :: time step 0049000 :: write_image_to_disk 0
[Rank 0000000, 3.024657e+03 s, 00000178815968 kB] :: time step 0049200 :: write_image_to_disk 0
[Rank 0000000, 3.036313e+03 s, 00000178815968 kB] :: time step 0049400 :: write_image_to_disk 0
[Rank 0000000, 3.047973e+03 s, 00000178815968 kB] :: time step 0049600 :: write_image_to_disk 0
[Rank 0000000, 3.059637e+03 s, 00000178815968 kB] :: time step 0049800 :: write_image_to_disk 0
[Rank 0000000, 3.071284e+03 s, 00000178815968 kB] :: time step 0050000 :: write_image_to_disk 1
[Rank 0000000, 3.083198e+03 s, 00000179853504 kB] :: time step 0050200 :: write_image_to_disk 0
[Rank 0000000, 3.094848e+03 s, 00000179853504 kB] :: time step 0050400 :: write_image_to_disk 0
[Rank 0000000, 3.106507e+03 s, 00000179853504 kB] :: time step 0050600 :: write_image_to_disk 0
[Rank 0000000, 3.118167e+03 s, 00000179853504 kB] :: time step 0050800 :: write_image_to_disk 0
[Rank 0000000, 3.129827e+03 s, 00000179853504 kB] :: time step 0051000 :: write_image_to_disk 0
[Rank 0000000, 3.141742e+03 s, 00000180891040 kB] :: time step 0051200 :: write_image_to_disk 0
[Rank 0000000, 3.153403e+03 s, 00000180891040 kB] :: time step 0051400 :: write_image_to_disk 0
[Rank 0000000, 3.165063e+03 s, 00000180891040 kB] :: time step 0051600 :: write_image_to_disk 0
[Rank 0000000, 3.176727e+03 s, 00000180891040 kB] :: time step 0051800 :: write_image_to_disk 0
[Rank 0000000, 3.188382e+03 s, 00000180891040 kB] :: time step 0052000 :: write_image_to_disk 0
[Rank 0000000, 3.200299e+03 s, 00000181928576 kB] :: time step 0052200 :: write_image_to_disk 0
[Rank 0000000, 3.211956e+03 s, 00000181928576 kB] :: time step 0052400 :: write_image_to_disk 0
[Rank 0000000, 3.223621e+03 s, 00000181928576 kB] :: time step 0052600 :: write_image_to_disk 0
[Rank 0000000, 3.235284e+03 s, 00000181928576 kB] :: time step 0052800 :: write_image_to_disk 0
[Rank 0000000, 3.246938e+03 s, 00000181928576 kB] :: time step 0053000 :: write_image_to_disk 0
[Rank 0000000, 3.258844e+03 s, 00000182966112 kB] :: time step 0053200 :: write_image_to_disk 0
[Rank 0000000, 3.270502e+03 s, 00000182966112 kB] :: time step 0053400 :: write_image_to_disk 0
[Rank 0000000, 3.282164e+03 s, 00000182966112 kB] :: time step 0053600 :: write_image_to_disk 0
[Rank 0000000, 3.293828e+03 s, 00000182966112 kB] :: time step 0053800 :: write_image_to_disk 0
[Rank 0000000, 3.305486e+03 s, 00000182966112 kB] :: time step 0054000 :: write_image_to_disk 0
[Rank 0000000, 3.317403e+03 s, 00000184003648 kB] :: time step 0054200 :: write_image_to_disk 0
[Rank 0000000, 3.329060e+03 s, 00000184003648 kB] :: time step 0054400 :: write_image_to_disk 0
[Rank 0000000, 3.340712e+03 s, 00000184003648 kB] :: time step 0054600 :: write_image_to_disk 0
[Rank 0000000, 3.352376e+03 s, 00000184003648 kB] :: time step 0054800 :: write_image_to_disk 0
[Rank 0000000, 3.364038e+03 s, 00000184003648 kB] :: time step 0055000 :: write_image_to_disk 0
[Rank 0000000, 3.375958e+03 s, 00000185041184 kB] :: time step 0055200 :: write_image_to_disk 0
[Rank 0000000, 3.387618e+03 s, 00000185041184 kB] :: time step 0055400 :: write_image_to_disk 0
[Rank 0000000, 3.399276e+03 s, 00000185041184 kB] :: time step 0055600 :: write_image_to_disk 0
[Rank 0000000, 3.410935e+03 s, 00000185041184 kB] :: time step 0055800 :: write_image_to_disk 0
[Rank 0000000, 3.422597e+03 s, 00000185041184 kB] :: time step 0056000 :: write_image_to_disk 0
[Rank 0000000, 3.434512e+03 s, 00000186078720 kB] :: time step 0056200 :: write_image_to_disk 0
[Rank 0000000, 3.446175e+03 s, 00000186078720 kB] :: time step 0056400 :: write_image_to_disk 0
[Rank 0000000, 3.457835e+03 s, 00000186078720 kB] :: time step 0056600 :: write_image_to_disk 0
[Rank 0000000, 3.469497e+03 s, 00000186078720 kB] :: time step 0056800 :: write_image_to_disk 0
[Rank 0000000, 3.481164e+03 s, 00000186078720 kB] :: time step 0057000 :: write_image_to_disk 0
[Rank 0000000, 3.493087e+03 s, 00000187116256 kB] :: time step 0057200 :: write_image_to_disk 0
[Rank 0000000, 3.504746e+03 s, 00000187116256 kB] :: time step 0057400 :: write_image_to_disk 0
[Rank 0000000, 3.516407e+03 s, 00000187116256 kB] :: time step 0057600 :: write_image_to_disk 0
[Rank 0000000, 3.528065e+03 s, 00000187116256 kB] :: time step 0057800 :: write_image_to_disk 0
[Rank 0000000, 3.539725e+03 s, 00000187116256 kB] :: time step 0058000 :: write_image_to_disk 0
[Rank 0000000, 3.551629e+03 s, 00000188153792 kB] :: time step 0058200 :: write_image_to_disk 0
[Rank 0000000, 3.563282e+03 s, 00000188153792 kB] :: time step 0058400 :: write_image_to_disk 0
[Rank 0000000, 3.574940e+03 s, 00000188153792 kB] :: time step 0058600 :: write_image_to_disk 0
[Rank 0000000, 3.586602e+03 s, 00000188153792 kB] :: time step 0058800 :: write_image_to_disk 0
[Rank 0000000, 3.598262e+03 s, 00000188153792 kB] :: time step 0059000 :: write_image_to_disk 0
[Rank 0000000, 3.610162e+03 s, 00000189191328 kB] :: time step 0059200 :: write_image_to_disk 0
[Rank 0000000, 3.621816e+03 s, 00000189191328 kB] :: time step 0059400 :: write_image_to_disk 0
[Rank 0000000, 3.633475e+03 s, 00000189191328 kB] :: time step 0059600 :: write_image_to_disk 0
[Rank 0000000, 3.645143e+03 s, 00000189191328 kB] :: time step 0059800 :: write_image_to_disk 0
[Rank 0000000, 3.656805e+03 s, 00000189191328 kB] :: time step 0060000 :: write_image_to_disk 1
[Rank 0000000, 3.668706e+03 s, 00000190228864 kB] :: time step 0060200 :: write_image_to_disk 0
[Rank 0000000, 3.680367e+03 s, 00000190228864 kB] :: time step 0060400 :: write_image_to_disk 0
[Rank 0000000, 3.692034e+03 s, 00000190228864 kB] :: time step 0060600 :: write_image_to_disk 0
[Rank 0000000, 3.703700e+03 s, 00000190228864 kB] :: time step 0060800 :: write_image_to_disk 0
[Rank 0000000, 3.715364e+03 s, 00000190228864 kB] :: time step 0061000 :: write_image_to_disk 0
[Rank 0000000, 3.727270e+03 s, 00000191266400 kB] :: time step 0061200 :: write_image_to_disk 0
[Rank 0000000, 3.738938e+03 s, 00000191266400 kB] :: time step 0061400 :: write_image_to_disk 0
[Rank 0000000, 3.750598e+03 s, 00000191266400 kB] :: time step 0061600 :: write_image_to_disk 0
[Rank 0000000, 3.762254e+03 s, 00000191266400 kB] :: time step 0061800 :: write_image_to_disk 0
[Rank 0000000, 3.773910e+03 s, 00000191266400 kB] :: time step 0062000 :: write_image_to_disk 0
[Rank 0000000, 3.785807e+03 s, 00000192303936 kB] :: time step 0062200 :: write_image_to_disk 0
[Rank 0000000, 3.797465e+03 s, 00000192303936 kB] :: time step 0062400 :: write_image_to_disk 0
[Rank 0000000, 3.809117e+03 s, 00000192303936 kB] :: time step 0062600 :: write_image_to_disk 0
[Rank 0000000, 3.820766e+03 s, 00000192303936 kB] :: time step 0062800 :: write_image_to_disk 0
[Rank 0000000, 3.832419e+03 s, 00000192303936 kB] :: time step 0063000 :: write_image_to_disk 0
[Rank 0000000, 3.844325e+03 s, 00000193341472 kB] :: time step 0063200 :: write_image_to_disk 0
[Rank 0000000, 3.855984e+03 s, 00000193341472 kB] :: time step 0063400 :: write_image_to_disk 0
[Rank 0000000, 3.867648e+03 s, 00000193341472 kB] :: time step 0063600 :: write_image_to_disk 0
[Rank 0000000, 3.879310e+03 s, 00000193341472 kB] :: time step 0063800 :: write_image_to_disk 0
[Rank 0000000, 3.890969e+03 s, 00000193341472 kB] :: time step 0064000 :: write_image_to_disk 0
[Rank 0000000, 3.902866e+03 s, 00000194379008 kB] :: time step 0064200 :: write_image_to_disk 0
[Rank 0000000, 3.914531e+03 s, 00000194379008 kB] :: time step 0064400 :: write_image_to_disk 0
[Rank 0000000, 3.926191e+03 s, 00000194379008 kB] :: time step 0064600 :: write_image_to_disk 0
[Rank 0000000, 3.937858e+03 s, 00000194379008 kB] :: time step 0064800 :: write_image_to_disk 0
[Rank 0000000, 3.949514e+03 s, 00000194379008 kB] :: time step 0065000 :: write_image_to_disk 0
[Rank 0000000, 3.961410e+03 s, 00000195416544 kB] :: time step 0065200 :: write_image_to_disk 0
[Rank 0000000, 3.973078e+03 s, 00000195416544 kB] :: time step 0065400 :: write_image_to_disk 0
[Rank 0000000, 3.984731e+03 s, 00000195416544 kB] :: time step 0065600 :: write_image_to_disk 0
[Rank 0000000, 3.996385e+03 s, 00000195416544 kB] :: time step 0065800 :: write_image_to_disk 0
[Rank 0000000, 4.008036e+03 s, 00000195416544 kB] :: time step 0066000 :: write_image_to_disk 0
[Rank 0000000, 4.019935e+03 s, 00000196454080 kB] :: time step 0066200 :: write_image_to_disk 0
[Rank 0000000, 4.031598e+03 s, 00000196454080 kB] :: time step 0066400 :: write_image_to_disk 0
[Rank 0000000, 4.043263e+03 s, 00000196454080 kB] :: time step 0066600 :: write_image_to_disk 0
[Rank 0000000, 4.054918e+03 s, 00000196454080 kB] :: time step 0066800 :: write_image_to_disk 0
[Rank 0000000, 4.066575e+03 s, 00000196454080 kB] :: time step 0067000 :: write_image_to_disk 0
[Rank 0000000, 4.078466e+03 s, 00000197491616 kB] :: time step 0067200 :: write_image_to_disk 0
[Rank 0000000, 4.090131e+03 s, 00000197491616 kB] :: time step 0067400 :: write_image_to_disk 0
[Rank 0000000, 4.101789e+03 s, 00000197491616 kB] :: time step 0067600 :: write_image_to_disk 0
[Rank 0000000, 4.113455e+03 s, 00000197491616 kB] :: time step 0067800 :: write_image_to_disk 0
[Rank 0000000, 4.125115e+03 s, 00000197491616 kB] :: time step 0068000 :: write_image_to_disk 0
[Rank 0000000, 4.137014e+03 s, 00000198529152 kB] :: time step 0068200 :: write_image_to_disk 0
[Rank 0000000, 4.148665e+03 s, 00000198529152 kB] :: time step 0068400 :: write_image_to_disk 0
[Rank 0000000, 4.160327e+03 s, 00000198529152 kB] :: time step 0068600 :: write_image_to_disk 0
[Rank 0000000, 4.171983e+03 s, 00000198529152 kB] :: time step 0068800 :: write_image_to_disk 0
[Rank 0000000, 4.183641e+03 s, 00000198529152 kB] :: time step 0069000 :: write_image_to_disk 0
[Rank 0000000, 4.195542e+03 s, 00000199566688 kB] :: time step 0069200 :: write_image_to_disk 0
[Rank 0000000, 4.207194e+03 s, 00000199566688 kB] :: time step 0069400 :: write_image_to_disk 0
[Rank 0000000, 4.218846e+03 s, 00000199566688 kB] :: time step 0069600 :: write_image_to_disk 0
[Rank 0000000, 4.230503e+03 s, 00000199566688 kB] :: time step 0069800 :: write_image_to_disk 0
[Rank 0000000, 4.242161e+03 s, 00000199566688 kB] :: time step 0070000 :: write_image_to_disk 1
[Rank 0000000, 4.254058e+03 s, 00000200604224 kB] :: time step 0070200 :: write_image_to_disk 0
[Rank 0000000, 4.265711e+03 s, 00000200604224 kB] :: time step 0070400 :: write_image_to_disk 0
[Rank 0000000, 4.277368e+03 s, 00000200604224 kB] :: time step 0070600 :: write_image_to_disk 0
[Rank 0000000, 4.289013e+03 s, 00000200604224 kB] :: time step 0070800 :: write_image_to_disk 0
[Rank 0000000, 4.300667e+03 s, 00000200604224 kB] :: time step 0071000 :: write_image_to_disk 0
[Rank 0000000, 4.312559e+03 s, 00000201641760 kB] :: time step 0071200 :: write_image_to_disk 0
[Rank 0000000, 4.324216e+03 s, 00000201641760 kB] :: time step 0071400 :: write_image_to_disk 0
[Rank 0000000, 4.335875e+03 s, 00000201641760 kB] :: time step 0071600 :: write_image_to_disk 0
[Rank 0000000, 4.347536e+03 s, 00000201641760 kB] :: time step 0071800 :: write_image_to_disk 0
[Rank 0000000, 4.359194e+03 s, 00000201641760 kB] :: time step 0072000 :: write_image_to_disk 0
[Rank 0000000, 4.371089e+03 s, 00000202679296 kB] :: time step 0072200 :: write_image_to_disk 0
[Rank 0000000, 4.382747e+03 s, 00000202679296 kB] :: time step 0072400 :: write_image_to_disk 0
[Rank 0000000, 4.394398e+03 s, 00000202679296 kB] :: time step 0072600 :: write_image_to_disk 0
[Rank 0000000, 4.406054e+03 s, 00000202679296 kB] :: time step 0072800 :: write_image_to_disk 0
[Rank 0000000, 4.417707e+03 s, 00000202679296 kB] :: time step 0073000 :: write_image_to_disk 0
[Rank 0000000, 4.429603e+03 s, 00000203716832 kB] :: time step 0073200 :: write_image_to_disk 0
[Rank 0000000, 4.441255e+03 s, 00000203716832 kB] :: time step 0073400 :: write_image_to_disk 0
[Rank 0000000, 4.452908e+03 s, 00000203716832 kB] :: time step 0073600 :: write_image_to_disk 0
[Rank 0000000, 4.464566e+03 s, 00000203716832 kB] :: time step 0073800 :: write_image_to_disk 0
[Rank 0000000, 4.476223e+03 s, 00000203716832 kB] :: time step 0074000 :: write_image_to_disk 0
[Rank 0000000, 4.488108e+03 s, 00000204754368 kB] :: time step 0074200 :: write_image_to_disk 0
[Rank 0000000, 4.499761e+03 s, 00000204754368 kB] :: time step 0074400 :: write_image_to_disk 0
[Rank 0000000, 4.511423e+03 s, 00000204754368 kB] :: time step 0074600 :: write_image_to_disk 0
[Rank 0000000, 4.523082e+03 s, 00000204754368 kB] :: time step 0074800 :: write_image_to_disk 0
[Rank 0000000, 4.534739e+03 s, 00000204754368 kB] :: time step 0075000 :: write_image_to_disk 0
[Rank 0000000, 4.546638e+03 s, 00000205791904 kB] :: time step 0075200 :: write_image_to_disk 0
[Rank 0000000, 4.558301e+03 s, 00000205791904 kB] :: time step 0075400 :: write_image_to_disk 0
[Rank 0000000, 4.569954e+03 s, 00000205791904 kB] :: time step 0075600 :: write_image_to_disk 0
[Rank 0000000, 4.581622e+03 s, 00000205791904 kB] :: time step 0075800 :: write_image_to_disk 0
[Rank 0000000, 4.593279e+03 s, 00000205791904 kB] :: time step 0076000 :: write_image_to_disk 0
[Rank 0000000, 4.605174e+03 s, 00000206829440 kB] :: time step 0076200 :: write_image_to_disk 0
[Rank 0000000, 4.616833e+03 s, 00000206829440 kB] :: time step 0076400 :: write_image_to_disk 0
[Rank 0000000, 4.628494e+03 s, 00000206829440 kB] :: time step 0076600 :: write_image_to_disk 0
[Rank 0000000, 4.640153e+03 s, 00000206829440 kB] :: time step 0076800 :: write_image_to_disk 0
[Rank 0000000, 4.651812e+03 s, 00000206829440 kB] :: time step 0077000 :: write_image_to_disk 0
[Rank 0000000, 4.663705e+03 s, 00000207866976 kB] :: time step 0077200 :: write_image_to_disk 0
[Rank 0000000, 4.675357e+03 s, 00000207866976 kB] :: time step 0077400 :: write_image_to_disk 0
[Rank 0000000, 4.687023e+03 s, 00000207866976 kB] :: time step 0077600 :: write_image_to_disk 0
[Rank 0000000, 4.698684e+03 s, 00000207866976 kB] :: time step 0077800 :: write_image_to_disk 0
[Rank 0000000, 4.710335e+03 s, 00000207866976 kB] :: time step 0078000 :: write_image_to_disk 0
[Rank 0000000, 4.722239e+03 s, 00000208904512 kB] :: time step 0078200 :: write_image_to_disk 0
[Rank 0000000, 4.733897e+03 s, 00000208904512 kB] :: time step 0078400 :: write_image_to_disk 0
[Rank 0000000, 4.745549e+03 s, 00000208904512 kB] :: time step 0078600 :: write_image_to_disk 0
[Rank 0000000, 4.757205e+03 s, 00000208904512 kB] :: time step 0078800 :: write_image_to_disk 0
[Rank 0000000, 4.768860e+03 s, 00000208904512 kB] :: time step 0079000 :: write_image_to_disk 0
[Rank 0000000, 4.780759e+03 s, 00000209942048 kB] :: time step 0079200 :: write_image_to_disk 0
[Rank 0000000, 4.792420e+03 s, 00000209942048 kB] :: time step 0079400 :: write_image_to_disk 0
[Rank 0000000, 4.804079e+03 s, 00000209942048 kB] :: time step 0079600 :: write_image_to_disk 0
[Rank 0000000, 4.815740e+03 s, 00000209942048 kB] :: time step 0079800 :: write_image_to_disk 0
[Rank 0000000, 4.827395e+03 s, 00000209942048 kB] :: time step 0080000 :: write_image_to_disk 1
[Rank 0000000, 4.839290e+03 s, 00000210979584 kB] :: time step 0080200 :: write_image_to_disk 0
[Rank 0000000, 4.850944e+03 s, 00000210979584 kB] :: time step 0080400 :: write_image_to_disk 0
[Rank 0000000, 4.862598e+03 s, 00000210979584 kB] :: time step 0080600 :: write_image_to_disk 0
[Rank 0000000, 4.874265e+03 s, 00000210979584 kB] :: time step 0080800 :: write_image_to_disk 0
[Rank 0000000, 4.885927e+03 s, 00000210979584 kB] :: time step 0081000 :: write_image_to_disk 0
[Rank 0000000, 4.897817e+03 s, 00000212017120 kB] :: time step 0081200 :: write_image_to_disk 0
[Rank 0000000, 4.909468e+03 s, 00000212017120 kB] :: time step 0081400 :: write_image_to_disk 0
[Rank 0000000, 4.921134e+03 s, 00000212017120 kB] :: time step 0081600 :: write_image_to_disk 0
[Rank 0000000, 4.932794e+03 s, 00000212017120 kB] :: time step 0081800 :: write_image_to_disk 0
[Rank 0000000, 4.944450e+03 s, 00000212017120 kB] :: time step 0082000 :: write_image_to_disk 0
[Rank 0000000, 4.956342e+03 s, 00000213054656 kB] :: time step 0082200 :: write_image_to_disk 0
[Rank 0000000, 4.968006e+03 s, 00000213054656 kB] :: time step 0082400 :: write_image_to_disk 0
[Rank 0000000, 4.979664e+03 s, 00000213054656 kB] :: time step 0082600 :: write_image_to_disk 0
[Rank 0000000, 4.991318e+03 s, 00000213054656 kB] :: time step 0082800 :: write_image_to_disk 0
[Rank 0000000, 5.002971e+03 s, 00000213054656 kB] :: time step 0083000 :: write_image_to_disk 0
[Rank 0000000, 5.014873e+03 s, 00000214092192 kB] :: time step 0083200 :: write_image_to_disk 0
[Rank 0000000, 5.026533e+03 s, 00000214092192 kB] :: time step 0083400 :: write_image_to_disk 0
[Rank 0000000, 5.038198e+03 s, 00000214092192 kB] :: time step 0083600 :: write_image_to_disk 0
[Rank 0000000, 5.049862e+03 s, 00000214092192 kB] :: time step 0083800 :: write_image_to_disk 0
[Rank 0000000, 5.061529e+03 s, 00000214092192 kB] :: time step 0084000 :: write_image_to_disk 0
[Rank 0000000, 5.073438e+03 s, 00000215129728 kB] :: time step 0084200 :: write_image_to_disk 0
[Rank 0000000, 5.085100e+03 s, 00000215129728 kB] :: time step 0084400 :: write_image_to_disk 0
[Rank 0000000, 5.096763e+03 s, 00000215129728 kB] :: time step 0084600 :: write_image_to_disk 0
[Rank 0000000, 5.108417e+03 s, 00000215129728 kB] :: time step 0084800 :: write_image_to_disk 0
[Rank 0000000, 5.120069e+03 s, 00000215129728 kB] :: time step 0085000 :: write_image_to_disk 0
[Rank 0000000, 5.131983e+03 s, 00000216167264 kB] :: time step 0085200 :: write_image_to_disk 0
[Rank 0000000, 5.143642e+03 s, 00000216167264 kB] :: time step 0085400 :: write_image_to_disk 0
[Rank 0000000, 5.155304e+03 s, 00000216167264 kB] :: time step 0085600 :: write_image_to_disk 0
[Rank 0000000, 5.166962e+03 s, 00000216167264 kB] :: time step 0085800 :: write_image_to_disk 0
[Rank 0000000, 5.178621e+03 s, 00000216167264 kB] :: time step 0086000 :: write_image_to_disk 0
[Rank 0000000, 5.190534e+03 s, 00000217204800 kB] :: time step 0086200 :: write_image_to_disk 0
[Rank 0000000, 5.202186e+03 s, 00000217204800 kB] :: time step 0086400 :: write_image_to_disk 0
[Rank 0000000, 5.213849e+03 s, 00000217204800 kB] :: time step 0086600 :: write_image_to_disk 0
[Rank 0000000, 5.225507e+03 s, 00000217204800 kB] :: time step 0086800 :: write_image_to_disk 0
[Rank 0000000, 5.237168e+03 s, 00000217204800 kB] :: time step 0087000 :: write_image_to_disk 0
[Rank 0000000, 5.249085e+03 s, 00000218242336 kB] :: time step 0087200 :: write_image_to_disk 0
[Rank 0000000, 5.260743e+03 s, 00000218242336 kB] :: time step 0087400 :: write_image_to_disk 0
[Rank 0000000, 5.272401e+03 s, 00000218242336 kB] :: time step 0087600 :: write_image_to_disk 0
[Rank 0000000, 5.284065e+03 s, 00000218242336 kB] :: time step 0087800 :: write_image_to_disk 0
[Rank 0000000, 5.295718e+03 s, 00000218242336 kB] :: time step 0088000 :: write_image_to_disk 0
[Rank 0000000, 5.307608e+03 s, 00000219279872 kB] :: time step 0088200 :: write_image_to_disk 0
[Rank 0000000, 5.319267e+03 s, 00000219279872 kB] :: time step 0088400 :: write_image_to_disk 0
[Rank 0000000, 5.330928e+03 s, 00000219279872 kB] :: time step 0088600 :: write_image_to_disk 0
[Rank 0000000, 5.342589e+03 s, 00000219279872 kB] :: time step 0088800 :: write_image_to_disk 0
[Rank 0000000, 5.354245e+03 s, 00000219279872 kB] :: time step 0089000 :: write_image_to_disk 0
[Rank 0000000, 5.366127e+03 s, 00000220316128 kB] :: time step 0089200 :: write_image_to_disk 0
[Rank 0000000, 5.377795e+03 s, 00000220316128 kB] :: time step 0089400 :: write_image_to_disk 0
[Rank 0000000, 5.389465e+03 s, 00000220316128 kB] :: time step 0089600 :: write_image_to_disk 0
[Rank 0000000, 5.401123e+03 s, 00000220316128 kB] :: time step 0089800 :: write_image_to_disk 0
[Rank 0000000, 5.412793e+03 s, 00000220316128 kB] :: time step 0090000 :: write_image_to_disk 1
[Rank 0000000, 5.424672e+03 s, 00000221337360 kB] :: time step 0090200 :: write_image_to_disk 0
[Rank 0000000, 5.436336e+03 s, 00000221337360 kB] :: time step 0090400 :: write_image_to_disk 0
[Rank 0000000, 5.447988e+03 s, 00000221337360 kB] :: time step 0090600 :: write_image_to_disk 0
[Rank 0000000, 5.459656e+03 s, 00000221337360 kB] :: time step 0090800 :: write_image_to_disk 0
[Rank 0000000, 5.471316e+03 s, 00000221337360 kB] :: time step 0091000 :: write_image_to_disk 0
[Rank 0000000, 5.483206e+03 s, 00000222379168 kB] :: time step 0091200 :: write_image_to_disk 0
[Rank 0000000, 5.494863e+03 s, 00000222379168 kB] :: time step 0091400 :: write_image_to_disk 0
[Rank 0000000, 5.506517e+03 s, 00000222379168 kB] :: time step 0091600 :: write_image_to_disk 0
[Rank 0000000, 5.518172e+03 s, 00000222379168 kB] :: time step 0091800 :: write_image_to_disk 0
[Rank 0000000, 5.529826e+03 s, 00000222379168 kB] :: time step 0092000 :: write_image_to_disk 0
[Rank 0000000, 5.541724e+03 s, 00000223416704 kB] :: time step 0092200 :: write_image_to_disk 0
[Rank 0000000, 5.553387e+03 s, 00000223416704 kB] :: time step 0092400 :: write_image_to_disk 0
[Rank 0000000, 5.565046e+03 s, 00000223416704 kB] :: time step 0092600 :: write_image_to_disk 0
[Rank 0000000, 5.576716e+03 s, 00000223416704 kB] :: time step 0092800 :: write_image_to_disk 0
[Rank 0000000, 5.588371e+03 s, 00000223416704 kB] :: time step 0093000 :: write_image_to_disk 0
[Rank 0000000, 5.600266e+03 s, 00000224454240 kB] :: time step 0093200 :: write_image_to_disk 0
[Rank 0000000, 5.611923e+03 s, 00000224454240 kB] :: time step 0093400 :: write_image_to_disk 0
[Rank 0000000, 5.623582e+03 s, 00000224454240 kB] :: time step 0093600 :: write_image_to_disk 0
[Rank 0000000, 5.635238e+03 s, 00000224454240 kB] :: time step 0093800 :: write_image_to_disk 0
[Rank 0000000, 5.646893e+03 s, 00000224454240 kB] :: time step 0094000 :: write_image_to_disk 0
[Rank 0000000, 5.658788e+03 s, 00000225491776 kB] :: time step 0094200 :: write_image_to_disk 0
[Rank 0000000, 5.670453e+03 s, 00000225491776 kB] :: time step 0094400 :: write_image_to_disk 0
[Rank 0000000, 5.682114e+03 s, 00000225491776 kB] :: time step 0094600 :: write_image_to_disk 0
[Rank 0000000, 5.693779e+03 s, 00000225491776 kB] :: time step 0094800 :: write_image_to_disk 0
[Rank 0000000, 5.705432e+03 s, 00000225491776 kB] :: time step 0095000 :: write_image_to_disk 0
[Rank 0000000, 5.717315e+03 s, 00000226529312 kB] :: time step 0095200 :: write_image_to_disk 0
[Rank 0000000, 5.728976e+03 s, 00000226529312 kB] :: time step 0095400 :: write_image_to_disk 0
[Rank 0000000, 5.740635e+03 s, 00000226529312 kB] :: time step 0095600 :: write_image_to_disk 0
[Rank 0000000, 5.752294e+03 s, 00000226529312 kB] :: time step 0095800 :: write_image_to_disk 0
[Rank 0000000, 5.763951e+03 s, 00000226529312 kB] :: time step 0096000 :: write_image_to_disk 0
[Rank 0000000, 5.775836e+03 s, 00000227566848 kB] :: time step 0096200 :: write_image_to_disk 0
[Rank 0000000, 5.787496e+03 s, 00000227566848 kB] :: time step 0096400 :: write_image_to_disk 0
[Rank 0000000, 5.799155e+03 s, 00000227566848 kB] :: time step 0096600 :: write_image_to_disk 0
[Rank 0000000, 5.810812e+03 s, 00000227566848 kB] :: time step 0096800 :: write_image_to_disk 0
[Rank 0000000, 5.822474e+03 s, 00000227566848 kB] :: time step 0097000 :: write_image_to_disk 0
[Rank 0000000, 5.834355e+03 s, 00000228604384 kB] :: time step 0097200 :: write_image_to_disk 0
[Rank 0000000, 5.846015e+03 s, 00000228604384 kB] :: time step 0097400 :: write_image_to_disk 0
[Rank 0000000, 5.857672e+03 s, 00000228604384 kB] :: time step 0097600 :: write_image_to_disk 0
[Rank 0000000, 5.869335e+03 s, 00000228604384 kB] :: time step 0097800 :: write_image_to_disk 0
[Rank 0000000, 5.880994e+03 s, 00000228604384 kB] :: time step 0098000 :: write_image_to_disk 0
[Rank 0000000, 5.892878e+03 s, 00000229641920 kB] :: time step 0098200 :: write_image_to_disk 0
[Rank 0000000, 5.904538e+03 s, 00000229641920 kB] :: time step 0098400 :: write_image_to_disk 0
[Rank 0000000, 5.916190e+03 s, 00000229641920 kB] :: time step 0098600 :: write_image_to_disk 0
[Rank 0000000, 5.927844e+03 s, 00000229641920 kB] :: time step 0098800 :: write_image_to_disk 0
[Rank 0000000, 5.939504e+03 s, 00000229641920 kB] :: time step 0099000 :: write_image_to_disk 0
[Rank 0000000, 5.951386e+03 s, 00000230679456 kB] :: time step 0099200 :: write_image_to_disk 0
[Rank 0000000, 5.963045e+03 s, 00000230679456 kB] :: time step 0099400 :: write_image_to_disk 0
[Rank 0000000, 5.974706e+03 s, 00000230679456 kB] :: time step 0099600 :: write_image_to_disk 0
[Rank 0000000, 5.986368e+03 s, 00000230679456 kB] :: time step 0099800 :: write_image_to_disk 0
[Rank 0000000, 5.998022e+03 s, 00000230679456 kB] :: time step 0100000 :: write_image_to_disk 1
[Rank 0000000, 6.013405e+03 s, 00000231349080 kB] :: -------------------
[Rank 0000000, 6.013405e+03 s, 00000231349080 kB] :: SIMULATION FINISHED

real	100m15.979s
user	500m24.809s
sys	405m58.879s
