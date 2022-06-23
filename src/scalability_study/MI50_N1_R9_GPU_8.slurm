

======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK    MCLK     Fan   Perf  PwrCap  VRAM%  GPU%  
0    36.0c  33.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
1    34.0c  35.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
2    35.0c  32.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
3    34.0c  35.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
4    34.0c  36.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
5    34.0c  36.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
6    34.0c  33.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
7    34.0c  33.0W   930Mhz  1000Mhz  255%  auto  225.0W    0%   0%    
================================================================================
============================= End of ROCm SMI Log ==============================
[Rank 0000000, 3.388180e-03 s, 00000004037584 kB] :: INITIALISE
[Rank 0000000, 3.427544e-03 s, 00000004037584 kB] :: ----------
[Rank 0000000, 3.440920e-03 s, 00000004037584 kB] :: --> loading input and decomposing geometry
[Rank 0000000, 2.083428e-02 s, 00000004051756 kB] :: ----> opened data file ../../cases/FileSharingIZPK/CBM2019_Arteries_patched.gmy
[Rank 0000000, 2.088470e-02 s, 00000004051756 kB] :: ----> reading preamble
[Rank 0000000, 2.103370e-02 s, 00000004051756 kB] :: ----> reading header (start)
[Rank 0000000, 3.816656e+00 s, 00000004162596 kB] :: ----> reading header (end)
[Rank 0000000, 4.926606e+00 s, 00000004162592 kB] :: ----> non-empty blocks: 199393
[Rank 0000000, 4.926639e+00 s, 00000004162592 kB] ::           total blocks: 119617344
[Rank 0000000, 4.926652e+00 s, 00000004162592 kB] ::                  ratio: 0.001667
[Rank 0000000, 4.926666e+00 s, 00000004162592 kB] ::                  sites: 66401494
[Rank 0000000, 4.926678e+00 s, 00000004162592 kB] :: ----> blockInformation.size(): 199393
[Rank 0000000, 4.926690e+00 s, 00000004162592 kB] ::  fluidSitesOnEachBlock.size(): 0
[Rank 0000000, 4.926702e+00 s, 00000004162592 kB] ::           blockWeights.size(): 0
[Rank 0000000, 4.926714e+00 s, 00000004162592 kB] :: ----> is blockInformation.size() == nonEmptyBlocks? yes
[Rank 0000000, 4.926726e+00 s, 00000004162592 kB] :: ----> not optimising decomposition
[Rank 0000000, 4.929358e+00 s, 00000004162592 kB] :: ----> basic decomposition (start)
[Rank 0000001, 8.465127e+00 s, 00000000480552 kB] :: ----> load distribution: 0.000020
[Rank 0000000, 8.626785e+00 s, 00000004292700 kB] :: ----> basic decomposition (end)
[Rank 0000000, 8.626824e+00 s, 00000004292700 kB] :: ----> read blocks (start)
[Rank 0000001, 1.412416e+01 s, 00000000489456 kB] :: ----> blockInformation.size(): 199393
[Rank 0000008, 1.412417e+01 s, 00000000480432 kB] :: ----> blockInformation.size(): 25107
[Rank 0000002, 1.412418e+01 s, 00000000489664 kB] :: ----> blockInformation.size(): 199393
[Rank 0000004, 1.412417e+01 s, 00000000480716 kB] :: ----> blockInformation.size(): 25771
[Rank 0000003, 1.412419e+01 s, 00000000480356 kB] :: ----> blockInformation.size(): 25368
[Rank 0000005, 1.412415e+01 s, 00000000480804 kB] :: ----> blockInformation.size(): 25761
[Rank 0000007, 1.412418e+01 s, 00000000480788 kB] :: ----> blockInformation.size(): 25036
[Rank 0000006, 1.412418e+01 s, 00000000480436 kB] :: ----> blockInformation.size(): 25325
[Rank 0000000, 7.115407e+01 s, 00000024094648 kB] :: ----> read blocks (end)
[Rank 0000000, 7.493233e+01 s, 00000024073660 kB] :: --> lattice data
[Rank 0000000, 8.930685e+01 s, 00000060396152 kB] :: ----> gathering lattice information (start)
[Rank 0000000, 8.930694e+01 s, 00000060396152 kB] :: ----> gathering lattice information (end)
[Rank 0000000, 1.134424e+02 s, 00000070258896 kB] :: --> neighbouring data manager
[Rank 0000000, 1.134425e+02 s, 00000070258896 kB] :: --> lattice-Boltzmann model
Rank 0: Detected 8 CUDA Capable device(s)
Rank: 8, Iolet ID: 19, Frequency: 1 
Rank: 8, Iolet ID: 20, Frequency: 1 
Rank: 8, Iolet ID: 21, Frequency: 1 
Rank: 8, Iolet ID: 22, Frequency: 3 
Rank: 8, Iolet ID: 23, Frequency: 3 
Rank: 8, Iolet ID: 29, Frequency: 2 
Rank: 8, Iolet ID: 30, Frequency: 2 
Rank: 8, Iolet ID: 31, Frequency: 1 
Rank: 8, Iolet ID: 32, Frequency: 1 
Rank: 8, Iolet ID: 34, Frequency: 1 
Rank: 8, Iolet ID: 35, Frequency: 1 
Rank: 8, Iolet ID: 36, Frequency: 1 
Rank: 8, Iolet ID: 37, Frequency: 1 
Rank: 8, Iolet ID: 19, Frequency: 1 
Rank: 8, Iolet ID: 20, Frequency: 1 
Rank: 8, Iolet ID: 21, Frequency: 1 
Rank: 8, Iolet ID: 22, Frequency: 3 
Rank: 8, Iolet ID: 23, Frequency: 4 
Rank: 8, Iolet ID: 29, Frequency: 2 
Rank: 8, Iolet ID: 30, Frequency: 2 
Rank: 8, Iolet ID: 31, Frequency: 1 
Rank: 8, Iolet ID: 32, Frequency: 1 
Rank: 8, Iolet ID: 34, Frequency: 1 
Rank: 8, Iolet ID: 35, Frequency: 1 
Rank: 8, Iolet ID: 36, Frequency: 1 
Rank: 8, Iolet ID: 37, Frequency: 1 
INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 8 - Total requested global memory 2.81GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 2 - Total requested global memory 3.01GB 

Rank: 7, Iolet ID: 18, Frequency: 1 
Rank: 7, Iolet ID: 24, Frequency: 1 
Rank: 7, Iolet ID: 25, Frequency: 2 
Rank: 7, Iolet ID: 26, Frequency: 2 
Rank: 7, Iolet ID: 27, Frequency: 1 
Rank: 7, Iolet ID: 28, Frequency: 1 
Rank: 7, Iolet ID: 33, Frequency: 1 
Rank: 7, Iolet ID: 18, Frequency: 1 
Rank: 7, Iolet ID: 24, Frequency: 1 
Rank: 7, Iolet ID: 25, Frequency: 2 
Rank: 7, Iolet ID: 26, Frequency: 2 
Rank: 7, Iolet ID: 27, Frequency: 1 
Rank: 7, Iolet ID: 28, Frequency: 1 
Rank: 7, Iolet ID: 33, Frequency: 1 
INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 7 - Total requested global memory 3.71GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 1 - Total requested global memory 3.80GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 3 - Total requested global memory 3.85GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 6 - Total requested global memory 3.89GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 4 - Total requested global memory 4.69GB 

INDEED the Inlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
INDEED the Outlet Type of BCs is: NASHZEROTHORDERPRESSUREIOLET 
Rank: 5 - Total requested global memory 4.93GB 

[Rank 0000000, 1.227505e+02 s, 00000130687008 kB] :: -------------------
[Rank 0000000, 1.227505e+02 s, 00000130687008 kB] :: INITIALISE FINISHED
[Rank 0000000, 1.230605e+02 s, 00000130661080 kB] :: SIMULATION STARTING
[Rank 0000000, 1.230606e+02 s, 00000130661080 kB] :: -------------------
[Rank 0000000, 1.300387e+02 s, 00000130919976 kB] :: time step 0000200 :: write_image_to_disk 0
[Rank 0000000, 1.371330e+02 s, 00000130920652 kB] :: time step 0000400 :: write_image_to_disk 0
[Rank 0000000, 1.442241e+02 s, 00000130920996 kB] :: time step 0000600 :: write_image_to_disk 0
[Rank 0000000, 1.513158e+02 s, 00000130921312 kB] :: time step 0000800 :: write_image_to_disk 0
[Rank 0000000, 1.584089e+02 s, 00000130921312 kB] :: time step 0001000 :: write_image_to_disk 0
[Rank 0000000, 1.656338e+02 s, 00000131960596 kB] :: time step 0001200 :: write_image_to_disk 0
[Rank 0000000, 1.727263e+02 s, 00000131960596 kB] :: time step 0001400 :: write_image_to_disk 0
[Rank 0000000, 1.798202e+02 s, 00000131960596 kB] :: time step 0001600 :: write_image_to_disk 0
[Rank 0000000, 1.869151e+02 s, 00000131960596 kB] :: time step 0001800 :: write_image_to_disk 0
[Rank 0000000, 1.940071e+02 s, 00000131960596 kB] :: time step 0002000 :: write_image_to_disk 0
[Rank 0000000, 2.012358e+02 s, 00000132998180 kB] :: time step 0002200 :: write_image_to_disk 0
[Rank 0000000, 2.083287e+02 s, 00000132998180 kB] :: time step 0002400 :: write_image_to_disk 0
[Rank 0000000, 2.154217e+02 s, 00000132998180 kB] :: time step 0002600 :: write_image_to_disk 0
[Rank 0000000, 2.225220e+02 s, 00000132998180 kB] :: time step 0002800 :: write_image_to_disk 0
[Rank 0000000, 2.296141e+02 s, 00000132998180 kB] :: time step 0003000 :: write_image_to_disk 0
[Rank 0000000, 2.368393e+02 s, 00000134036456 kB] :: time step 0003200 :: write_image_to_disk 0
[Rank 0000000, 2.439339e+02 s, 00000134036456 kB] :: time step 0003400 :: write_image_to_disk 0
[Rank 0000000, 2.510271e+02 s, 00000134036456 kB] :: time step 0003600 :: write_image_to_disk 0
[Rank 0000000, 2.581235e+02 s, 00000134036456 kB] :: time step 0003800 :: write_image_to_disk 0
[Rank 0000000, 2.652172e+02 s, 00000134036456 kB] :: time step 0004000 :: write_image_to_disk 0
[Rank 0000000, 2.724457e+02 s, 00000135074100 kB] :: time step 0004200 :: write_image_to_disk 0
[Rank 0000000, 2.795397e+02 s, 00000135074100 kB] :: time step 0004400 :: write_image_to_disk 0
[Rank 0000000, 2.866327e+02 s, 00000135074100 kB] :: time step 0004600 :: write_image_to_disk 0
[Rank 0000000, 2.937283e+02 s, 00000135074100 kB] :: time step 0004800 :: write_image_to_disk 0
[Rank 0000000, 3.008243e+02 s, 00000135074100 kB] :: time step 0005000 :: write_image_to_disk 0
[Rank 0000000, 3.080542e+02 s, 00000136111664 kB] :: time step 0005200 :: write_image_to_disk 0
[Rank 0000000, 3.151501e+02 s, 00000136111664 kB] :: time step 0005400 :: write_image_to_disk 0
[Rank 0000000, 3.222430e+02 s, 00000136111664 kB] :: time step 0005600 :: write_image_to_disk 0
[Rank 0000000, 3.293368e+02 s, 00000136111664 kB] :: time step 0005800 :: write_image_to_disk 0
[Rank 0000000, 3.364304e+02 s, 00000136111664 kB] :: time step 0006000 :: write_image_to_disk 0
[Rank 0000000, 3.436624e+02 s, 00000137149232 kB] :: time step 0006200 :: write_image_to_disk 0
[Rank 0000000, 3.507615e+02 s, 00000137149232 kB] :: time step 0006400 :: write_image_to_disk 0
[Rank 0000000, 3.578559e+02 s, 00000137149232 kB] :: time step 0006600 :: write_image_to_disk 0
[Rank 0000000, 3.649519e+02 s, 00000137149232 kB] :: time step 0006800 :: write_image_to_disk 0
[Rank 0000000, 3.720483e+02 s, 00000137149232 kB] :: time step 0007000 :: write_image_to_disk 0
[Rank 0000000, 3.792803e+02 s, 00000138186580 kB] :: time step 0007200 :: write_image_to_disk 0
[Rank 0000000, 3.863779e+02 s, 00000138186580 kB] :: time step 0007400 :: write_image_to_disk 0
[Rank 0000000, 3.934765e+02 s, 00000138186580 kB] :: time step 0007600 :: write_image_to_disk 0
[Rank 0000000, 4.005744e+02 s, 00000138186580 kB] :: time step 0007800 :: write_image_to_disk 0
[Rank 0000000, 4.076710e+02 s, 00000138186580 kB] :: time step 0008000 :: write_image_to_disk 0
[Rank 0000000, 4.149061e+02 s, 00000139224376 kB] :: time step 0008200 :: write_image_to_disk 0
[Rank 0000000, 4.220049e+02 s, 00000139224376 kB] :: time step 0008400 :: write_image_to_disk 0
[Rank 0000000, 4.291012e+02 s, 00000139224376 kB] :: time step 0008600 :: write_image_to_disk 0
[Rank 0000000, 4.361993e+02 s, 00000139224376 kB] :: time step 0008800 :: write_image_to_disk 0
[Rank 0000000, 4.432942e+02 s, 00000139224376 kB] :: time step 0009000 :: write_image_to_disk 0
[Rank 0000000, 4.505274e+02 s, 00000140261944 kB] :: time step 0009200 :: write_image_to_disk 0
[Rank 0000000, 4.576279e+02 s, 00000140261944 kB] :: time step 0009400 :: write_image_to_disk 0
[Rank 0000000, 4.647258e+02 s, 00000140261944 kB] :: time step 0009600 :: write_image_to_disk 0
[Rank 0000000, 4.718246e+02 s, 00000140261944 kB] :: time step 0009800 :: write_image_to_disk 0
[Rank 0000000, 4.789244e+02 s, 00000140261944 kB] :: time step 0010000 :: write_image_to_disk 1
[Rank 0000000, 4.861598e+02 s, 00000141299516 kB] :: time step 0010200 :: write_image_to_disk 0
[Rank 0000000, 4.932581e+02 s, 00000141299516 kB] :: time step 0010400 :: write_image_to_disk 0
[Rank 0000000, 5.003579e+02 s, 00000141299516 kB] :: time step 0010600 :: write_image_to_disk 0
[Rank 0000000, 5.074570e+02 s, 00000141299516 kB] :: time step 0010800 :: write_image_to_disk 0
[Rank 0000000, 5.145575e+02 s, 00000141299516 kB] :: time step 0011000 :: write_image_to_disk 0
[Rank 0000000, 5.217957e+02 s, 00000142337092 kB] :: time step 0011200 :: write_image_to_disk 0
[Rank 0000000, 5.288968e+02 s, 00000142337092 kB] :: time step 0011400 :: write_image_to_disk 0
[Rank 0000000, 5.359952e+02 s, 00000142337092 kB] :: time step 0011600 :: write_image_to_disk 0
[Rank 0000000, 5.430962e+02 s, 00000142337092 kB] :: time step 0011800 :: write_image_to_disk 0
[Rank 0000000, 5.501953e+02 s, 00000142337092 kB] :: time step 0012000 :: write_image_to_disk 0
[Rank 0000000, 5.574328e+02 s, 00000143374432 kB] :: time step 0012200 :: write_image_to_disk 0
[Rank 0000000, 5.645321e+02 s, 00000143374432 kB] :: time step 0012400 :: write_image_to_disk 0
[Rank 0000000, 5.716315e+02 s, 00000143374432 kB] :: time step 0012600 :: write_image_to_disk 0
[Rank 0000000, 5.787315e+02 s, 00000143374432 kB] :: time step 0012800 :: write_image_to_disk 0
[Rank 0000000, 5.858320e+02 s, 00000143374432 kB] :: time step 0013000 :: write_image_to_disk 0
[Rank 0000000, 5.930659e+02 s, 00000144412228 kB] :: time step 0013200 :: write_image_to_disk 0
[Rank 0000000, 6.001653e+02 s, 00000144412228 kB] :: time step 0013400 :: write_image_to_disk 0
[Rank 0000000, 6.072646e+02 s, 00000144412228 kB] :: time step 0013600 :: write_image_to_disk 0
[Rank 0000000, 6.143640e+02 s, 00000144412228 kB] :: time step 0013800 :: write_image_to_disk 0
[Rank 0000000, 6.214624e+02 s, 00000144412228 kB] :: time step 0014000 :: write_image_to_disk 0
[Rank 0000000, 6.286967e+02 s, 00000145449552 kB] :: time step 0014200 :: write_image_to_disk 0
[Rank 0000000, 6.357971e+02 s, 00000145449552 kB] :: time step 0014400 :: write_image_to_disk 0
[Rank 0000000, 6.428958e+02 s, 00000145449552 kB] :: time step 0014600 :: write_image_to_disk 0
[Rank 0000000, 6.499950e+02 s, 00000145449552 kB] :: time step 0014800 :: write_image_to_disk 0
[Rank 0000000, 6.570942e+02 s, 00000145449552 kB] :: time step 0015000 :: write_image_to_disk 0
[Rank 0000000, 6.643257e+02 s, 00000146487148 kB] :: time step 0015200 :: write_image_to_disk 0
[Rank 0000000, 6.714257e+02 s, 00000146487148 kB] :: time step 0015400 :: write_image_to_disk 0
[Rank 0000000, 6.785243e+02 s, 00000146487148 kB] :: time step 0015600 :: write_image_to_disk 0
[Rank 0000000, 6.856217e+02 s, 00000146487148 kB] :: time step 0015800 :: write_image_to_disk 0
[Rank 0000000, 6.927217e+02 s, 00000146487148 kB] :: time step 0016000 :: write_image_to_disk 0
[Rank 0000000, 6.999552e+02 s, 00000147524748 kB] :: time step 0016200 :: write_image_to_disk 0
[Rank 0000000, 7.070544e+02 s, 00000147524748 kB] :: time step 0016400 :: write_image_to_disk 0
[Rank 0000000, 7.141529e+02 s, 00000147524748 kB] :: time step 0016600 :: write_image_to_disk 0
[Rank 0000000, 7.212501e+02 s, 00000147524748 kB] :: time step 0016800 :: write_image_to_disk 0
[Rank 0000000, 7.283494e+02 s, 00000147524748 kB] :: time step 0017000 :: write_image_to_disk 0
[Rank 0000000, 7.355812e+02 s, 00000148562352 kB] :: time step 0017200 :: write_image_to_disk 0
[Rank 0000000, 7.426810e+02 s, 00000148562352 kB] :: time step 0017400 :: write_image_to_disk 0
[Rank 0000000, 7.497873e+02 s, 00000148562352 kB] :: time step 0017600 :: write_image_to_disk 0
[Rank 0000000, 7.568864e+02 s, 00000148562352 kB] :: time step 0017800 :: write_image_to_disk 0
[Rank 0000000, 7.639850e+02 s, 00000148562352 kB] :: time step 0018000 :: write_image_to_disk 0
[Rank 0000000, 7.712170e+02 s, 00000149599952 kB] :: time step 0018200 :: write_image_to_disk 0
[Rank 0000000, 7.783163e+02 s, 00000149599952 kB] :: time step 0018400 :: write_image_to_disk 0
[Rank 0000000, 7.854157e+02 s, 00000149599952 kB] :: time step 0018600 :: write_image_to_disk 0
[Rank 0000000, 7.925136e+02 s, 00000149599952 kB] :: time step 0018800 :: write_image_to_disk 0
[Rank 0000000, 7.996167e+02 s, 00000149599952 kB] :: time step 0019000 :: write_image_to_disk 0
[Rank 0000000, 8.068505e+02 s, 00000150637544 kB] :: time step 0019200 :: write_image_to_disk 0
[Rank 0000000, 8.139495e+02 s, 00000150637544 kB] :: time step 0019400 :: write_image_to_disk 0
[Rank 0000000, 8.210481e+02 s, 00000150637544 kB] :: time step 0019600 :: write_image_to_disk 0
[Rank 0000000, 8.281471e+02 s, 00000150637544 kB] :: time step 0019800 :: write_image_to_disk 0
[Rank 0000000, 8.352446e+02 s, 00000150637544 kB] :: time step 0020000 :: write_image_to_disk 1
[Rank 0000000, 8.424794e+02 s, 00000151675140 kB] :: time step 0020200 :: write_image_to_disk 0
[Rank 0000000, 8.495776e+02 s, 00000151675140 kB] :: time step 0020400 :: write_image_to_disk 0
[Rank 0000000, 8.566758e+02 s, 00000151675140 kB] :: time step 0020600 :: write_image_to_disk 0
[Rank 0000000, 8.637777e+02 s, 00000151675140 kB] :: time step 0020800 :: write_image_to_disk 0
[Rank 0000000, 8.708768e+02 s, 00000151675140 kB] :: time step 0021000 :: write_image_to_disk 0
[Rank 0000000, 8.781095e+02 s, 00000152712468 kB] :: time step 0021200 :: write_image_to_disk 0
[Rank 0000000, 8.852071e+02 s, 00000152712468 kB] :: time step 0021400 :: write_image_to_disk 0
[Rank 0000000, 8.923058e+02 s, 00000152712468 kB] :: time step 0021600 :: write_image_to_disk 0
[Rank 0000000, 8.994052e+02 s, 00000152712468 kB] :: time step 0021800 :: write_image_to_disk 0
[Rank 0000000, 9.065042e+02 s, 00000152712468 kB] :: time step 0022000 :: write_image_to_disk 0
[Rank 0000000, 9.137405e+02 s, 00000153750064 kB] :: time step 0022200 :: write_image_to_disk 0
[Rank 0000000, 9.208395e+02 s, 00000153750064 kB] :: time step 0022400 :: write_image_to_disk 0
[Rank 0000000, 9.279374e+02 s, 00000153750064 kB] :: time step 0022600 :: write_image_to_disk 0
[Rank 0000000, 9.350364e+02 s, 00000153750064 kB] :: time step 0022800 :: write_image_to_disk 0
[Rank 0000000, 9.421341e+02 s, 00000153750064 kB] :: time step 0023000 :: write_image_to_disk 0
[Rank 0000000, 9.493672e+02 s, 00000154787656 kB] :: time step 0023200 :: write_image_to_disk 0
[Rank 0000000, 9.564651e+02 s, 00000154787656 kB] :: time step 0023400 :: write_image_to_disk 0
[Rank 0000000, 9.635638e+02 s, 00000154787656 kB] :: time step 0023600 :: write_image_to_disk 0
[Rank 0000000, 9.706653e+02 s, 00000154787656 kB] :: time step 0023800 :: write_image_to_disk 0
[Rank 0000000, 9.777642e+02 s, 00000154787656 kB] :: time step 0024000 :: write_image_to_disk 0
[Rank 0000000, 9.849980e+02 s, 00000155825256 kB] :: time step 0024200 :: write_image_to_disk 0
[Rank 0000000, 9.920991e+02 s, 00000155825256 kB] :: time step 0024400 :: write_image_to_disk 0
[Rank 0000000, 9.991957e+02 s, 00000155825256 kB] :: time step 0024600 :: write_image_to_disk 0
[Rank 0000000, 1.006294e+03 s, 00000155825256 kB] :: time step 0024800 :: write_image_to_disk 0
[Rank 0000000, 1.013393e+03 s, 00000155825256 kB] :: time step 0025000 :: write_image_to_disk 0
[Rank 0000000, 1.020627e+03 s, 00000156862852 kB] :: time step 0025200 :: write_image_to_disk 0
[Rank 0000000, 1.027729e+03 s, 00000156862852 kB] :: time step 0025400 :: write_image_to_disk 0
[Rank 0000000, 1.034827e+03 s, 00000156862852 kB] :: time step 0025600 :: write_image_to_disk 0
[Rank 0000000, 1.041928e+03 s, 00000156862852 kB] :: time step 0025800 :: write_image_to_disk 0
[Rank 0000000, 1.049026e+03 s, 00000156862852 kB] :: time step 0026000 :: write_image_to_disk 0
[Rank 0000000, 1.056262e+03 s, 00000157900440 kB] :: time step 0026200 :: write_image_to_disk 0
[Rank 0000000, 1.063359e+03 s, 00000157900440 kB] :: time step 0026400 :: write_image_to_disk 0
[Rank 0000000, 1.070457e+03 s, 00000157900440 kB] :: time step 0026600 :: write_image_to_disk 0
[Rank 0000000, 1.077555e+03 s, 00000157900440 kB] :: time step 0026800 :: write_image_to_disk 0
[Rank 0000000, 1.084652e+03 s, 00000157900440 kB] :: time step 0027000 :: write_image_to_disk 0
[Rank 0000000, 1.091887e+03 s, 00000158938032 kB] :: time step 0027200 :: write_image_to_disk 0
[Rank 0000000, 1.098985e+03 s, 00000158938032 kB] :: time step 0027400 :: write_image_to_disk 0
[Rank 0000000, 1.106086e+03 s, 00000158938032 kB] :: time step 0027600 :: write_image_to_disk 0
[Rank 0000000, 1.113183e+03 s, 00000158938032 kB] :: time step 0027800 :: write_image_to_disk 0
[Rank 0000000, 1.120280e+03 s, 00000158938032 kB] :: time step 0028000 :: write_image_to_disk 0
[Rank 0000000, 1.127516e+03 s, 00000159975360 kB] :: time step 0028200 :: write_image_to_disk 0
[Rank 0000000, 1.134615e+03 s, 00000159975360 kB] :: time step 0028400 :: write_image_to_disk 0
[Rank 0000000, 1.141715e+03 s, 00000159975360 kB] :: time step 0028600 :: write_image_to_disk 0
[Rank 0000000, 1.148813e+03 s, 00000159975360 kB] :: time step 0028800 :: write_image_to_disk 0
[Rank 0000000, 1.155910e+03 s, 00000159975360 kB] :: time step 0029000 :: write_image_to_disk 0
[Rank 0000000, 1.163143e+03 s, 00000161012956 kB] :: time step 0029200 :: write_image_to_disk 0
[Rank 0000000, 1.170241e+03 s, 00000161012956 kB] :: time step 0029400 :: write_image_to_disk 0
[Rank 0000000, 1.177338e+03 s, 00000161012956 kB] :: time step 0029600 :: write_image_to_disk 0
[Rank 0000000, 1.184437e+03 s, 00000161012956 kB] :: time step 0029800 :: write_image_to_disk 0
[Rank 0000000, 1.191537e+03 s, 00000161012956 kB] :: time step 0030000 :: write_image_to_disk 1
[Rank 0000000, 1.198772e+03 s, 00000162050552 kB] :: time step 0030200 :: write_image_to_disk 0
[Rank 0000000, 1.205869e+03 s, 00000162050552 kB] :: time step 0030400 :: write_image_to_disk 0
[Rank 0000000, 1.212967e+03 s, 00000162050552 kB] :: time step 0030600 :: write_image_to_disk 0
[Rank 0000000, 1.220064e+03 s, 00000162050552 kB] :: time step 0030800 :: write_image_to_disk 0
[Rank 0000000, 1.227166e+03 s, 00000162050552 kB] :: time step 0031000 :: write_image_to_disk 0
[Rank 0000000, 1.234403e+03 s, 00000163088140 kB] :: time step 0031200 :: write_image_to_disk 0
[Rank 0000000, 1.241502e+03 s, 00000163088140 kB] :: time step 0031400 :: write_image_to_disk 0
[Rank 0000000, 1.248599e+03 s, 00000163088140 kB] :: time step 0031600 :: write_image_to_disk 0
[Rank 0000000, 1.255700e+03 s, 00000163088140 kB] :: time step 0031800 :: write_image_to_disk 0
[Rank 0000000, 1.262796e+03 s, 00000163088140 kB] :: time step 0032000 :: write_image_to_disk 0
[Rank 0000000, 1.270032e+03 s, 00000164125728 kB] :: time step 0032200 :: write_image_to_disk 0
[Rank 0000000, 1.277132e+03 s, 00000164125728 kB] :: time step 0032400 :: write_image_to_disk 0
[Rank 0000000, 1.284229e+03 s, 00000164125728 kB] :: time step 0032600 :: write_image_to_disk 0
[Rank 0000000, 1.291327e+03 s, 00000164125728 kB] :: time step 0032800 :: write_image_to_disk 0
[Rank 0000000, 1.298426e+03 s, 00000164125728 kB] :: time step 0033000 :: write_image_to_disk 0
[Rank 0000000, 1.305660e+03 s, 00000165163324 kB] :: time step 0033200 :: write_image_to_disk 0
[Rank 0000000, 1.312758e+03 s, 00000165163324 kB] :: time step 0033400 :: write_image_to_disk 0
[Rank 0000000, 1.319857e+03 s, 00000165163324 kB] :: time step 0033600 :: write_image_to_disk 0
[Rank 0000000, 1.326957e+03 s, 00000165163324 kB] :: time step 0033800 :: write_image_to_disk 0
[Rank 0000000, 1.334055e+03 s, 00000165163324 kB] :: time step 0034000 :: write_image_to_disk 0
[Rank 0000000, 1.341290e+03 s, 00000166200912 kB] :: time step 0034200 :: write_image_to_disk 0
[Rank 0000000, 1.348387e+03 s, 00000166200912 kB] :: time step 0034400 :: write_image_to_disk 0
[Rank 0000000, 1.355486e+03 s, 00000166200912 kB] :: time step 0034600 :: write_image_to_disk 0
[Rank 0000000, 1.362585e+03 s, 00000166200912 kB] :: time step 0034800 :: write_image_to_disk 0
[Rank 0000000, 1.369684e+03 s, 00000166200912 kB] :: time step 0035000 :: write_image_to_disk 0
[Rank 0000000, 1.376922e+03 s, 00000167238500 kB] :: time step 0035200 :: write_image_to_disk 0
[Rank 0000000, 1.384021e+03 s, 00000167238500 kB] :: time step 0035400 :: write_image_to_disk 0
[Rank 0000000, 1.391122e+03 s, 00000167238500 kB] :: time step 0035600 :: write_image_to_disk 0
[Rank 0000000, 1.398221e+03 s, 00000167238500 kB] :: time step 0035800 :: write_image_to_disk 0
[Rank 0000000, 1.405318e+03 s, 00000167238500 kB] :: time step 0036000 :: write_image_to_disk 0
[Rank 0000000, 1.412556e+03 s, 00000168275824 kB] :: time step 0036200 :: write_image_to_disk 0
[Rank 0000000, 1.419656e+03 s, 00000168275824 kB] :: time step 0036400 :: write_image_to_disk 0
[Rank 0000000, 1.426755e+03 s, 00000168275824 kB] :: time step 0036600 :: write_image_to_disk 0
[Rank 0000000, 1.433853e+03 s, 00000168275824 kB] :: time step 0036800 :: write_image_to_disk 0
[Rank 0000000, 1.440952e+03 s, 00000168275824 kB] :: time step 0037000 :: write_image_to_disk 0
[Rank 0000000, 1.448190e+03 s, 00000169313412 kB] :: time step 0037200 :: write_image_to_disk 0
[Rank 0000000, 1.455289e+03 s, 00000169313412 kB] :: time step 0037400 :: write_image_to_disk 0
[Rank 0000000, 1.462387e+03 s, 00000169313412 kB] :: time step 0037600 :: write_image_to_disk 0
[Rank 0000000, 1.469487e+03 s, 00000169313412 kB] :: time step 0037800 :: write_image_to_disk 0
[Rank 0000000, 1.476585e+03 s, 00000169313412 kB] :: time step 0038000 :: write_image_to_disk 0
[Rank 0000000, 1.483821e+03 s, 00000170351000 kB] :: time step 0038200 :: write_image_to_disk 0
[Rank 0000000, 1.490922e+03 s, 00000170351000 kB] :: time step 0038400 :: write_image_to_disk 0
[Rank 0000000, 1.498021e+03 s, 00000170351000 kB] :: time step 0038600 :: write_image_to_disk 0
[Rank 0000000, 1.505122e+03 s, 00000170351000 kB] :: time step 0038800 :: write_image_to_disk 0
[Rank 0000000, 1.512220e+03 s, 00000170351000 kB] :: time step 0039000 :: write_image_to_disk 0
[Rank 0000000, 1.519456e+03 s, 00000171388588 kB] :: time step 0039200 :: write_image_to_disk 0
[Rank 0000000, 1.526557e+03 s, 00000171388588 kB] :: time step 0039400 :: write_image_to_disk 0
[Rank 0000000, 1.533656e+03 s, 00000171388588 kB] :: time step 0039600 :: write_image_to_disk 0
[Rank 0000000, 1.540754e+03 s, 00000171388588 kB] :: time step 0039800 :: write_image_to_disk 0
[Rank 0000000, 1.547851e+03 s, 00000171388588 kB] :: time step 0040000 :: write_image_to_disk 1
[Rank 0000000, 1.555087e+03 s, 00000172426180 kB] :: time step 0040200 :: write_image_to_disk 0
[Rank 0000000, 1.562186e+03 s, 00000172426180 kB] :: time step 0040400 :: write_image_to_disk 0
[Rank 0000000, 1.569290e+03 s, 00000172426180 kB] :: time step 0040600 :: write_image_to_disk 0
[Rank 0000000, 1.576387e+03 s, 00000172426180 kB] :: time step 0040800 :: write_image_to_disk 0
[Rank 0000000, 1.583484e+03 s, 00000172426180 kB] :: time step 0041000 :: write_image_to_disk 0
[Rank 0000000, 1.590722e+03 s, 00000173463768 kB] :: time step 0041200 :: write_image_to_disk 0
[Rank 0000000, 1.597822e+03 s, 00000173463768 kB] :: time step 0041400 :: write_image_to_disk 0
[Rank 0000000, 1.604921e+03 s, 00000173463768 kB] :: time step 0041600 :: write_image_to_disk 0
[Rank 0000000, 1.612019e+03 s, 00000173463768 kB] :: time step 0041800 :: write_image_to_disk 0
[Rank 0000000, 1.619119e+03 s, 00000173463768 kB] :: time step 0042000 :: write_image_to_disk 0
[Rank 0000000, 1.626352e+03 s, 00000174501356 kB] :: time step 0042200 :: write_image_to_disk 0
[Rank 0000000, 1.633452e+03 s, 00000174501356 kB] :: time step 0042400 :: write_image_to_disk 0
[Rank 0000000, 1.640549e+03 s, 00000174501356 kB] :: time step 0042600 :: write_image_to_disk 0
[Rank 0000000, 1.647646e+03 s, 00000174501356 kB] :: time step 0042800 :: write_image_to_disk 0
[Rank 0000000, 1.654745e+03 s, 00000174501356 kB] :: time step 0043000 :: write_image_to_disk 0
[Rank 0000000, 1.661982e+03 s, 00000175538680 kB] :: time step 0043200 :: write_image_to_disk 0
[Rank 0000000, 1.669081e+03 s, 00000175538680 kB] :: time step 0043400 :: write_image_to_disk 0
[Rank 0000000, 1.676181e+03 s, 00000175538680 kB] :: time step 0043600 :: write_image_to_disk 0
[Rank 0000000, 1.683280e+03 s, 00000175538680 kB] :: time step 0043800 :: write_image_to_disk 0
[Rank 0000000, 1.690378e+03 s, 00000175538680 kB] :: time step 0044000 :: write_image_to_disk 0
[Rank 0000000, 1.697615e+03 s, 00000176576276 kB] :: time step 0044200 :: write_image_to_disk 0
[Rank 0000000, 1.704714e+03 s, 00000176576276 kB] :: time step 0044400 :: write_image_to_disk 0
[Rank 0000000, 1.711814e+03 s, 00000176576276 kB] :: time step 0044600 :: write_image_to_disk 0
[Rank 0000000, 1.718912e+03 s, 00000176576276 kB] :: time step 0044800 :: write_image_to_disk 0
[Rank 0000000, 1.726012e+03 s, 00000176576276 kB] :: time step 0045000 :: write_image_to_disk 0
[Rank 0000000, 1.733253e+03 s, 00000177613864 kB] :: time step 0045200 :: write_image_to_disk 0
[Rank 0000000, 1.740352e+03 s, 00000177613864 kB] :: time step 0045400 :: write_image_to_disk 0
[Rank 0000000, 1.747454e+03 s, 00000177613864 kB] :: time step 0045600 :: write_image_to_disk 0
[Rank 0000000, 1.754552e+03 s, 00000177613864 kB] :: time step 0045800 :: write_image_to_disk 0
[Rank 0000000, 1.761652e+03 s, 00000177613864 kB] :: time step 0046000 :: write_image_to_disk 0
[Rank 0000000, 1.768903e+03 s, 00000178651452 kB] :: time step 0046200 :: write_image_to_disk 0
[Rank 0000000, 1.776004e+03 s, 00000178651452 kB] :: time step 0046400 :: write_image_to_disk 0
[Rank 0000000, 1.783106e+03 s, 00000178651452 kB] :: time step 0046600 :: write_image_to_disk 0
[Rank 0000000, 1.790207e+03 s, 00000178651452 kB] :: time step 0046800 :: write_image_to_disk 0
[Rank 0000000, 1.797307e+03 s, 00000178651452 kB] :: time step 0047000 :: write_image_to_disk 0
[Rank 0000000, 1.804547e+03 s, 00000179689048 kB] :: time step 0047200 :: write_image_to_disk 0
[Rank 0000000, 1.811649e+03 s, 00000179689048 kB] :: time step 0047400 :: write_image_to_disk 0
[Rank 0000000, 1.818747e+03 s, 00000179689048 kB] :: time step 0047600 :: write_image_to_disk 0
[Rank 0000000, 1.825849e+03 s, 00000179689048 kB] :: time step 0047800 :: write_image_to_disk 0
[Rank 0000000, 1.832948e+03 s, 00000179689048 kB] :: time step 0048000 :: write_image_to_disk 0
[Rank 0000000, 1.840187e+03 s, 00000180726640 kB] :: time step 0048200 :: write_image_to_disk 0
[Rank 0000000, 1.847288e+03 s, 00000180726640 kB] :: time step 0048400 :: write_image_to_disk 0
[Rank 0000000, 1.854390e+03 s, 00000180726640 kB] :: time step 0048600 :: write_image_to_disk 0
[Rank 0000000, 1.861489e+03 s, 00000180726640 kB] :: time step 0048800 :: write_image_to_disk 0
[Rank 0000000, 1.868590e+03 s, 00000180726640 kB] :: time step 0049000 :: write_image_to_disk 0
[Rank 0000000, 1.875863e+03 s, 00000181764228 kB] :: time step 0049200 :: write_image_to_disk 0
[Rank 0000000, 1.882965e+03 s, 00000181764228 kB] :: time step 0049400 :: write_image_to_disk 0
[Rank 0000000, 1.890067e+03 s, 00000181764228 kB] :: time step 0049600 :: write_image_to_disk 0
[Rank 0000000, 1.897169e+03 s, 00000181764228 kB] :: time step 0049800 :: write_image_to_disk 0
[Rank 0000000, 1.904271e+03 s, 00000181764228 kB] :: time step 0050000 :: write_image_to_disk 1
[Rank 0000000, 1.911525e+03 s, 00000182801552 kB] :: time step 0050200 :: write_image_to_disk 0
[Rank 0000000, 1.918625e+03 s, 00000182801552 kB] :: time step 0050400 :: write_image_to_disk 0
[Rank 0000000, 1.925727e+03 s, 00000182801552 kB] :: time step 0050600 :: write_image_to_disk 0
[Rank 0000000, 1.932827e+03 s, 00000182801552 kB] :: time step 0050800 :: write_image_to_disk 0
[Rank 0000000, 1.939928e+03 s, 00000182801552 kB] :: time step 0051000 :: write_image_to_disk 0
[Rank 0000000, 1.947182e+03 s, 00000183839140 kB] :: time step 0051200 :: write_image_to_disk 0
[Rank 0000000, 1.954283e+03 s, 00000183839140 kB] :: time step 0051400 :: write_image_to_disk 0
[Rank 0000000, 1.961384e+03 s, 00000183839140 kB] :: time step 0051600 :: write_image_to_disk 0
[Rank 0000000, 1.968485e+03 s, 00000183839140 kB] :: time step 0051800 :: write_image_to_disk 0
[Rank 0000000, 1.975588e+03 s, 00000183839140 kB] :: time step 0052000 :: write_image_to_disk 0
[Rank 0000000, 1.982839e+03 s, 00000184876728 kB] :: time step 0052200 :: write_image_to_disk 0
[Rank 0000000, 1.989941e+03 s, 00000184876728 kB] :: time step 0052400 :: write_image_to_disk 0
[Rank 0000000, 1.997045e+03 s, 00000184876728 kB] :: time step 0052600 :: write_image_to_disk 0
[Rank 0000000, 2.004146e+03 s, 00000184876728 kB] :: time step 0052800 :: write_image_to_disk 0
[Rank 0000000, 2.011245e+03 s, 00000184876728 kB] :: time step 0053000 :: write_image_to_disk 0
[Rank 0000000, 2.018494e+03 s, 00000185914316 kB] :: time step 0053200 :: write_image_to_disk 0
[Rank 0000000, 2.025597e+03 s, 00000185914316 kB] :: time step 0053400 :: write_image_to_disk 0
[Rank 0000000, 2.032697e+03 s, 00000185914316 kB] :: time step 0053600 :: write_image_to_disk 0
[Rank 0000000, 2.039797e+03 s, 00000185914316 kB] :: time step 0053800 :: write_image_to_disk 0
[Rank 0000000, 2.046897e+03 s, 00000185914316 kB] :: time step 0054000 :: write_image_to_disk 0
[Rank 0000000, 2.054145e+03 s, 00000186951904 kB] :: time step 0054200 :: write_image_to_disk 0
[Rank 0000000, 2.061246e+03 s, 00000186951904 kB] :: time step 0054400 :: write_image_to_disk 0
[Rank 0000000, 2.068349e+03 s, 00000186951904 kB] :: time step 0054600 :: write_image_to_disk 0
[Rank 0000000, 2.075452e+03 s, 00000186951904 kB] :: time step 0054800 :: write_image_to_disk 0
[Rank 0000000, 2.082551e+03 s, 00000186951904 kB] :: time step 0055000 :: write_image_to_disk 0
[Rank 0000000, 2.089800e+03 s, 00000187989492 kB] :: time step 0055200 :: write_image_to_disk 0
[Rank 0000000, 2.096899e+03 s, 00000187989492 kB] :: time step 0055400 :: write_image_to_disk 0
[Rank 0000000, 2.104000e+03 s, 00000187989492 kB] :: time step 0055600 :: write_image_to_disk 0
[Rank 0000000, 2.111100e+03 s, 00000187989492 kB] :: time step 0055800 :: write_image_to_disk 0
[Rank 0000000, 2.118199e+03 s, 00000187989492 kB] :: time step 0056000 :: write_image_to_disk 0
[Rank 0000000, 2.125449e+03 s, 00000189027084 kB] :: time step 0056200 :: write_image_to_disk 0
[Rank 0000000, 2.132548e+03 s, 00000189027084 kB] :: time step 0056400 :: write_image_to_disk 0
[Rank 0000000, 2.139646e+03 s, 00000189027084 kB] :: time step 0056600 :: write_image_to_disk 0
[Rank 0000000, 2.146745e+03 s, 00000189027084 kB] :: time step 0056800 :: write_image_to_disk 0
[Rank 0000000, 2.153844e+03 s, 00000189027084 kB] :: time step 0057000 :: write_image_to_disk 0
[Rank 0000000, 2.161093e+03 s, 00000190064672 kB] :: time step 0057200 :: write_image_to_disk 0
[Rank 0000000, 2.168192e+03 s, 00000190064672 kB] :: time step 0057400 :: write_image_to_disk 0
[Rank 0000000, 2.175293e+03 s, 00000190064672 kB] :: time step 0057600 :: write_image_to_disk 0
[Rank 0000000, 2.182392e+03 s, 00000190064672 kB] :: time step 0057800 :: write_image_to_disk 0
[Rank 0000000, 2.189493e+03 s, 00000190064672 kB] :: time step 0058000 :: write_image_to_disk 0
[Rank 0000000, 2.196741e+03 s, 00000191101996 kB] :: time step 0058200 :: write_image_to_disk 0
[Rank 0000000, 2.203841e+03 s, 00000191101996 kB] :: time step 0058400 :: write_image_to_disk 0
[Rank 0000000, 2.210939e+03 s, 00000191101996 kB] :: time step 0058600 :: write_image_to_disk 0
[Rank 0000000, 2.218037e+03 s, 00000191101996 kB] :: time step 0058800 :: write_image_to_disk 0
[Rank 0000000, 2.225135e+03 s, 00000191101996 kB] :: time step 0059000 :: write_image_to_disk 0
[Rank 0000000, 2.232386e+03 s, 00000192139584 kB] :: time step 0059200 :: write_image_to_disk 0
[Rank 0000000, 2.239494e+03 s, 00000192139584 kB] :: time step 0059400 :: write_image_to_disk 0
[Rank 0000000, 2.246592e+03 s, 00000192139584 kB] :: time step 0059600 :: write_image_to_disk 0
[Rank 0000000, 2.253690e+03 s, 00000192139584 kB] :: time step 0059800 :: write_image_to_disk 0
[Rank 0000000, 2.260787e+03 s, 00000192139584 kB] :: time step 0060000 :: write_image_to_disk 1
[Rank 0000000, 2.268034e+03 s, 00000193177172 kB] :: time step 0060200 :: write_image_to_disk 0
[Rank 0000000, 2.275134e+03 s, 00000193177172 kB] :: time step 0060400 :: write_image_to_disk 0
[Rank 0000000, 2.282236e+03 s, 00000193177172 kB] :: time step 0060600 :: write_image_to_disk 0
[Rank 0000000, 2.289337e+03 s, 00000193177172 kB] :: time step 0060800 :: write_image_to_disk 0
[Rank 0000000, 2.296438e+03 s, 00000193177172 kB] :: time step 0061000 :: write_image_to_disk 0
[Rank 0000000, 2.303688e+03 s, 00000194214760 kB] :: time step 0061200 :: write_image_to_disk 0
[Rank 0000000, 2.310786e+03 s, 00000194214760 kB] :: time step 0061400 :: write_image_to_disk 0
[Rank 0000000, 2.317884e+03 s, 00000194214760 kB] :: time step 0061600 :: write_image_to_disk 0
[Rank 0000000, 2.324983e+03 s, 00000194214760 kB] :: time step 0061800 :: write_image_to_disk 0
[Rank 0000000, 2.332080e+03 s, 00000194214760 kB] :: time step 0062000 :: write_image_to_disk 0
[Rank 0000000, 2.339326e+03 s, 00000195252348 kB] :: time step 0062200 :: write_image_to_disk 0
[Rank 0000000, 2.346424e+03 s, 00000195252348 kB] :: time step 0062400 :: write_image_to_disk 0
[Rank 0000000, 2.353523e+03 s, 00000195252348 kB] :: time step 0062600 :: write_image_to_disk 0
[Rank 0000000, 2.360623e+03 s, 00000195252348 kB] :: time step 0062800 :: write_image_to_disk 0
[Rank 0000000, 2.367720e+03 s, 00000195252348 kB] :: time step 0063000 :: write_image_to_disk 0
[Rank 0000000, 2.374967e+03 s, 00000196289936 kB] :: time step 0063200 :: write_image_to_disk 0
[Rank 0000000, 2.382066e+03 s, 00000196289936 kB] :: time step 0063400 :: write_image_to_disk 0
[Rank 0000000, 2.389164e+03 s, 00000196289936 kB] :: time step 0063600 :: write_image_to_disk 0
[Rank 0000000, 2.396263e+03 s, 00000196289936 kB] :: time step 0063800 :: write_image_to_disk 0
[Rank 0000000, 2.403362e+03 s, 00000196289936 kB] :: time step 0064000 :: write_image_to_disk 0
[Rank 0000000, 2.410614e+03 s, 00000197327524 kB] :: time step 0064200 :: write_image_to_disk 0
[Rank 0000000, 2.417712e+03 s, 00000197327524 kB] :: time step 0064400 :: write_image_to_disk 0
[Rank 0000000, 2.424812e+03 s, 00000197327524 kB] :: time step 0064600 :: write_image_to_disk 0
[Rank 0000000, 2.431909e+03 s, 00000197327524 kB] :: time step 0064800 :: write_image_to_disk 0
[Rank 0000000, 2.439011e+03 s, 00000197327524 kB] :: time step 0065000 :: write_image_to_disk 0
[Rank 0000000, 2.446260e+03 s, 00000198364852 kB] :: time step 0065200 :: write_image_to_disk 0
[Rank 0000000, 2.453360e+03 s, 00000198364852 kB] :: time step 0065400 :: write_image_to_disk 0
[Rank 0000000, 2.460461e+03 s, 00000198364852 kB] :: time step 0065600 :: write_image_to_disk 0
[Rank 0000000, 2.467558e+03 s, 00000198364852 kB] :: time step 0065800 :: write_image_to_disk 0
[Rank 0000000, 2.474657e+03 s, 00000198364852 kB] :: time step 0066000 :: write_image_to_disk 0
[Rank 0000000, 2.481902e+03 s, 00000199402440 kB] :: time step 0066200 :: write_image_to_disk 0
[Rank 0000000, 2.489001e+03 s, 00000199402440 kB] :: time step 0066400 :: write_image_to_disk 0
[Rank 0000000, 2.496099e+03 s, 00000199402440 kB] :: time step 0066600 :: write_image_to_disk 0
[Rank 0000000, 2.503197e+03 s, 00000199402440 kB] :: time step 0066800 :: write_image_to_disk 0
[Rank 0000000, 2.510300e+03 s, 00000199402440 kB] :: time step 0067000 :: write_image_to_disk 0
[Rank 0000000, 2.517547e+03 s, 00000200440028 kB] :: time step 0067200 :: write_image_to_disk 0
[Rank 0000000, 2.524645e+03 s, 00000200440028 kB] :: time step 0067400 :: write_image_to_disk 0
[Rank 0000000, 2.531744e+03 s, 00000200440028 kB] :: time step 0067600 :: write_image_to_disk 0
[Rank 0000000, 2.538843e+03 s, 00000200440028 kB] :: time step 0067800 :: write_image_to_disk 0
[Rank 0000000, 2.545941e+03 s, 00000200440028 kB] :: time step 0068000 :: write_image_to_disk 0
[Rank 0000000, 2.553188e+03 s, 00000201477616 kB] :: time step 0068200 :: write_image_to_disk 0
[Rank 0000000, 2.560287e+03 s, 00000201477616 kB] :: time step 0068400 :: write_image_to_disk 0
[Rank 0000000, 2.567386e+03 s, 00000201477616 kB] :: time step 0068600 :: write_image_to_disk 0
[Rank 0000000, 2.574485e+03 s, 00000201477616 kB] :: time step 0068800 :: write_image_to_disk 0
[Rank 0000000, 2.581584e+03 s, 00000201477616 kB] :: time step 0069000 :: write_image_to_disk 0
[Rank 0000000, 2.588828e+03 s, 00000202515204 kB] :: time step 0069200 :: write_image_to_disk 0
[Rank 0000000, 2.595927e+03 s, 00000202515204 kB] :: time step 0069400 :: write_image_to_disk 0
[Rank 0000000, 2.603027e+03 s, 00000202515204 kB] :: time step 0069600 :: write_image_to_disk 0
[Rank 0000000, 2.610124e+03 s, 00000202515204 kB] :: time step 0069800 :: write_image_to_disk 0
[Rank 0000000, 2.617224e+03 s, 00000202515204 kB] :: time step 0070000 :: write_image_to_disk 1
[Rank 0000000, 2.624467e+03 s, 00000203552792 kB] :: time step 0070200 :: write_image_to_disk 0
[Rank 0000000, 2.631563e+03 s, 00000203552792 kB] :: time step 0070400 :: write_image_to_disk 0
[Rank 0000000, 2.638660e+03 s, 00000203552792 kB] :: time step 0070600 :: write_image_to_disk 0
[Rank 0000000, 2.645757e+03 s, 00000203552792 kB] :: time step 0070800 :: write_image_to_disk 0
[Rank 0000000, 2.652857e+03 s, 00000203552792 kB] :: time step 0071000 :: write_image_to_disk 0
[Rank 0000000, 2.660100e+03 s, 00000204590380 kB] :: time step 0071200 :: write_image_to_disk 0
[Rank 0000000, 2.667198e+03 s, 00000204590380 kB] :: time step 0071400 :: write_image_to_disk 0
[Rank 0000000, 2.674296e+03 s, 00000204590380 kB] :: time step 0071600 :: write_image_to_disk 0
[Rank 0000000, 2.681393e+03 s, 00000204590380 kB] :: time step 0071800 :: write_image_to_disk 0
[Rank 0000000, 2.688491e+03 s, 00000204590380 kB] :: time step 0072000 :: write_image_to_disk 0
[Rank 0000000, 2.695733e+03 s, 00000205627704 kB] :: time step 0072200 :: write_image_to_disk 0
[Rank 0000000, 2.702829e+03 s, 00000205627704 kB] :: time step 0072400 :: write_image_to_disk 0
[Rank 0000000, 2.709926e+03 s, 00000205627704 kB] :: time step 0072600 :: write_image_to_disk 0
[Rank 0000000, 2.717022e+03 s, 00000205627704 kB] :: time step 0072800 :: write_image_to_disk 0
[Rank 0000000, 2.724118e+03 s, 00000205627704 kB] :: time step 0073000 :: write_image_to_disk 0
[Rank 0000000, 2.731360e+03 s, 00000206665296 kB] :: time step 0073200 :: write_image_to_disk 0
[Rank 0000000, 2.738456e+03 s, 00000206665296 kB] :: time step 0073400 :: write_image_to_disk 0
[Rank 0000000, 2.745556e+03 s, 00000206665296 kB] :: time step 0073600 :: write_image_to_disk 0
[Rank 0000000, 2.752656e+03 s, 00000206665296 kB] :: time step 0073800 :: write_image_to_disk 0
[Rank 0000000, 2.759756e+03 s, 00000206665296 kB] :: time step 0074000 :: write_image_to_disk 0
[Rank 0000000, 2.766998e+03 s, 00000207702884 kB] :: time step 0074200 :: write_image_to_disk 0
[Rank 0000000, 2.774095e+03 s, 00000207702884 kB] :: time step 0074400 :: write_image_to_disk 0
[Rank 0000000, 2.781191e+03 s, 00000207702884 kB] :: time step 0074600 :: write_image_to_disk 0
[Rank 0000000, 2.788289e+03 s, 00000207702884 kB] :: time step 0074800 :: write_image_to_disk 0
[Rank 0000000, 2.795387e+03 s, 00000207702884 kB] :: time step 0075000 :: write_image_to_disk 0
[Rank 0000000, 2.802638e+03 s, 00000208740476 kB] :: time step 0075200 :: write_image_to_disk 0
[Rank 0000000, 2.809735e+03 s, 00000208740476 kB] :: time step 0075400 :: write_image_to_disk 0
[Rank 0000000, 2.816835e+03 s, 00000208740476 kB] :: time step 0075600 :: write_image_to_disk 0
[Rank 0000000, 2.823931e+03 s, 00000208740476 kB] :: time step 0075800 :: write_image_to_disk 0
[Rank 0000000, 2.831030e+03 s, 00000208740476 kB] :: time step 0076000 :: write_image_to_disk 0
[Rank 0000000, 2.838285e+03 s, 00000209778064 kB] :: time step 0076200 :: write_image_to_disk 0
[Rank 0000000, 2.845380e+03 s, 00000209778064 kB] :: time step 0076400 :: write_image_to_disk 0
[Rank 0000000, 2.852476e+03 s, 00000209778064 kB] :: time step 0076600 :: write_image_to_disk 0
[Rank 0000000, 2.859571e+03 s, 00000209778064 kB] :: time step 0076800 :: write_image_to_disk 0
[Rank 0000000, 2.866671e+03 s, 00000209778064 kB] :: time step 0077000 :: write_image_to_disk 0
[Rank 0000000, 2.873925e+03 s, 00000210815652 kB] :: time step 0077200 :: write_image_to_disk 0
[Rank 0000000, 2.881023e+03 s, 00000210815652 kB] :: time step 0077400 :: write_image_to_disk 0
[Rank 0000000, 2.888120e+03 s, 00000210815652 kB] :: time step 0077600 :: write_image_to_disk 0
[Rank 0000000, 2.895220e+03 s, 00000210815652 kB] :: time step 0077800 :: write_image_to_disk 0
[Rank 0000000, 2.902320e+03 s, 00000210815652 kB] :: time step 0078000 :: write_image_to_disk 0
[Rank 0000000, 2.909573e+03 s, 00000211853240 kB] :: time step 0078200 :: write_image_to_disk 0
[Rank 0000000, 2.916673e+03 s, 00000211853240 kB] :: time step 0078400 :: write_image_to_disk 0
[Rank 0000000, 2.923772e+03 s, 00000211853240 kB] :: time step 0078600 :: write_image_to_disk 0
[Rank 0000000, 2.930869e+03 s, 00000211853240 kB] :: time step 0078800 :: write_image_to_disk 0
[Rank 0000000, 2.937966e+03 s, 00000211853240 kB] :: time step 0079000 :: write_image_to_disk 0
[Rank 0000000, 2.945212e+03 s, 00000212890828 kB] :: time step 0079200 :: write_image_to_disk 0
[Rank 0000000, 2.952310e+03 s, 00000212890828 kB] :: time step 0079400 :: write_image_to_disk 0
[Rank 0000000, 2.959409e+03 s, 00000212890828 kB] :: time step 0079600 :: write_image_to_disk 0
[Rank 0000000, 2.966508e+03 s, 00000212890828 kB] :: time step 0079800 :: write_image_to_disk 0
[Rank 0000000, 2.973606e+03 s, 00000212890828 kB] :: time step 0080000 :: write_image_to_disk 1
[Rank 0000000, 2.980852e+03 s, 00000213928152 kB] :: time step 0080200 :: write_image_to_disk 0
[Rank 0000000, 2.987952e+03 s, 00000213928152 kB] :: time step 0080400 :: write_image_to_disk 0
[Rank 0000000, 2.995048e+03 s, 00000213928152 kB] :: time step 0080600 :: write_image_to_disk 0
[Rank 0000000, 3.002146e+03 s, 00000213928152 kB] :: time step 0080800 :: write_image_to_disk 0
[Rank 0000000, 3.009244e+03 s, 00000213928152 kB] :: time step 0081000 :: write_image_to_disk 0
[Rank 0000000, 3.016489e+03 s, 00000214965740 kB] :: time step 0081200 :: write_image_to_disk 0
[Rank 0000000, 3.023590e+03 s, 00000214965740 kB] :: time step 0081400 :: write_image_to_disk 0
[Rank 0000000, 3.030688e+03 s, 00000214965740 kB] :: time step 0081600 :: write_image_to_disk 0
[Rank 0000000, 3.037788e+03 s, 00000214965740 kB] :: time step 0081800 :: write_image_to_disk 0
[Rank 0000000, 3.044888e+03 s, 00000214965740 kB] :: time step 0082000 :: write_image_to_disk 0
[Rank 0000000, 3.052136e+03 s, 00000216003328 kB] :: time step 0082200 :: write_image_to_disk 0
[Rank 0000000, 3.059235e+03 s, 00000216003328 kB] :: time step 0082400 :: write_image_to_disk 0
[Rank 0000000, 3.066334e+03 s, 00000216003328 kB] :: time step 0082600 :: write_image_to_disk 0
[Rank 0000000, 3.073433e+03 s, 00000216003328 kB] :: time step 0082800 :: write_image_to_disk 0
[Rank 0000000, 3.080533e+03 s, 00000216003328 kB] :: time step 0083000 :: write_image_to_disk 0
[Rank 0000000, 3.087776e+03 s, 00000217040920 kB] :: time step 0083200 :: write_image_to_disk 0
[Rank 0000000, 3.094876e+03 s, 00000217040920 kB] :: time step 0083400 :: write_image_to_disk 0
[Rank 0000000, 3.101976e+03 s, 00000217040920 kB] :: time step 0083600 :: write_image_to_disk 0
[Rank 0000000, 3.109074e+03 s, 00000217040920 kB] :: time step 0083800 :: write_image_to_disk 0
[Rank 0000000, 3.116172e+03 s, 00000217040920 kB] :: time step 0084000 :: write_image_to_disk 0
[Rank 0000000, 3.123418e+03 s, 00000218078508 kB] :: time step 0084200 :: write_image_to_disk 0
[Rank 0000000, 3.130518e+03 s, 00000218078508 kB] :: time step 0084400 :: write_image_to_disk 0
[Rank 0000000, 3.137617e+03 s, 00000218078508 kB] :: time step 0084600 :: write_image_to_disk 0
[Rank 0000000, 3.144715e+03 s, 00000218078508 kB] :: time step 0084800 :: write_image_to_disk 0
[Rank 0000000, 3.151813e+03 s, 00000218078508 kB] :: time step 0085000 :: write_image_to_disk 0
[Rank 0000000, 3.159056e+03 s, 00000219116096 kB] :: time step 0085200 :: write_image_to_disk 0
[Rank 0000000, 3.166157e+03 s, 00000219116096 kB] :: time step 0085400 :: write_image_to_disk 0
[Rank 0000000, 3.173254e+03 s, 00000219116096 kB] :: time step 0085600 :: write_image_to_disk 0
[Rank 0000000, 3.180354e+03 s, 00000219116096 kB] :: time step 0085800 :: write_image_to_disk 0
[Rank 0000000, 3.187453e+03 s, 00000219116096 kB] :: time step 0086000 :: write_image_to_disk 0
[Rank 0000000, 3.194696e+03 s, 00000220153684 kB] :: time step 0086200 :: write_image_to_disk 0
[Rank 0000000, 3.201801e+03 s, 00000220153684 kB] :: time step 0086400 :: write_image_to_disk 0
[Rank 0000000, 3.208900e+03 s, 00000220153684 kB] :: time step 0086600 :: write_image_to_disk 0
[Rank 0000000, 3.215997e+03 s, 00000220153684 kB] :: time step 0086800 :: write_image_to_disk 0
[Rank 0000000, 3.223096e+03 s, 00000220153684 kB] :: time step 0087000 :: write_image_to_disk 0
[Rank 0000000, 3.230348e+03 s, 00000221191008 kB] :: time step 0087200 :: write_image_to_disk 0
[Rank 0000000, 3.237447e+03 s, 00000221191008 kB] :: time step 0087400 :: write_image_to_disk 0
[Rank 0000000, 3.244546e+03 s, 00000221191008 kB] :: time step 0087600 :: write_image_to_disk 0
[Rank 0000000, 3.251645e+03 s, 00000221191008 kB] :: time step 0087800 :: write_image_to_disk 0
[Rank 0000000, 3.258744e+03 s, 00000221191008 kB] :: time step 0088000 :: write_image_to_disk 0
[Rank 0000000, 3.265987e+03 s, 00000222228596 kB] :: time step 0088200 :: write_image_to_disk 0
[Rank 0000000, 3.273085e+03 s, 00000222228596 kB] :: time step 0088400 :: write_image_to_disk 0
[Rank 0000000, 3.280183e+03 s, 00000222228596 kB] :: time step 0088600 :: write_image_to_disk 0
[Rank 0000000, 3.287282e+03 s, 00000222228596 kB] :: time step 0088800 :: write_image_to_disk 0
[Rank 0000000, 3.294380e+03 s, 00000222228596 kB] :: time step 0089000 :: write_image_to_disk 0
[Rank 0000000, 3.301624e+03 s, 00000223266184 kB] :: time step 0089200 :: write_image_to_disk 0
[Rank 0000000, 3.308724e+03 s, 00000223266184 kB] :: time step 0089400 :: write_image_to_disk 0
[Rank 0000000, 3.315821e+03 s, 00000223266184 kB] :: time step 0089600 :: write_image_to_disk 0
[Rank 0000000, 3.322917e+03 s, 00000223266184 kB] :: time step 0089800 :: write_image_to_disk 0
[Rank 0000000, 3.330015e+03 s, 00000223266184 kB] :: time step 0090000 :: write_image_to_disk 1
[Rank 0000000, 3.337257e+03 s, 00000224303772 kB] :: time step 0090200 :: write_image_to_disk 0
[Rank 0000000, 3.344354e+03 s, 00000224303772 kB] :: time step 0090400 :: write_image_to_disk 0
[Rank 0000000, 3.351451e+03 s, 00000224303772 kB] :: time step 0090600 :: write_image_to_disk 0
[Rank 0000000, 3.358548e+03 s, 00000224303772 kB] :: time step 0090800 :: write_image_to_disk 0
[Rank 0000000, 3.365646e+03 s, 00000224303772 kB] :: time step 0091000 :: write_image_to_disk 0
[Rank 0000000, 3.372883e+03 s, 00000225341360 kB] :: time step 0091200 :: write_image_to_disk 0
[Rank 0000000, 3.379980e+03 s, 00000225341360 kB] :: time step 0091400 :: write_image_to_disk 0
[Rank 0000000, 3.387075e+03 s, 00000225341360 kB] :: time step 0091600 :: write_image_to_disk 0
[Rank 0000000, 3.394172e+03 s, 00000225341360 kB] :: time step 0091800 :: write_image_to_disk 0
[Rank 0000000, 3.401269e+03 s, 00000225341360 kB] :: time step 0092000 :: write_image_to_disk 0
[Rank 0000000, 3.408505e+03 s, 00000226378948 kB] :: time step 0092200 :: write_image_to_disk 0
[Rank 0000000, 3.415603e+03 s, 00000226378948 kB] :: time step 0092400 :: write_image_to_disk 0
[Rank 0000000, 3.422699e+03 s, 00000226378948 kB] :: time step 0092600 :: write_image_to_disk 0
[Rank 0000000, 3.429800e+03 s, 00000226378948 kB] :: time step 0092800 :: write_image_to_disk 0
[Rank 0000000, 3.436897e+03 s, 00000226378948 kB] :: time step 0093000 :: write_image_to_disk 0
[Rank 0000000, 3.444136e+03 s, 00000227416536 kB] :: time step 0093200 :: write_image_to_disk 0
[Rank 0000000, 3.451235e+03 s, 00000227416536 kB] :: time step 0093400 :: write_image_to_disk 0
[Rank 0000000, 3.458333e+03 s, 00000227416536 kB] :: time step 0093600 :: write_image_to_disk 0
[Rank 0000000, 3.465431e+03 s, 00000227416536 kB] :: time step 0093800 :: write_image_to_disk 0
[Rank 0000000, 3.472526e+03 s, 00000227416536 kB] :: time step 0094000 :: write_image_to_disk 0
[Rank 0000000, 3.479767e+03 s, 00000228453860 kB] :: time step 0094200 :: write_image_to_disk 0
[Rank 0000000, 3.486864e+03 s, 00000228453860 kB] :: time step 0094400 :: write_image_to_disk 0
[Rank 0000000, 3.493962e+03 s, 00000228453860 kB] :: time step 0094600 :: write_image_to_disk 0
[Rank 0000000, 3.501065e+03 s, 00000228453860 kB] :: time step 0094800 :: write_image_to_disk 0
[Rank 0000000, 3.508163e+03 s, 00000228453860 kB] :: time step 0095000 :: write_image_to_disk 0
[Rank 0000000, 3.515400e+03 s, 00000229491448 kB] :: time step 0095200 :: write_image_to_disk 0
[Rank 0000000, 3.522498e+03 s, 00000229491448 kB] :: time step 0095400 :: write_image_to_disk 0
[Rank 0000000, 3.529601e+03 s, 00000229491448 kB] :: time step 0095600 :: write_image_to_disk 0
[Rank 0000000, 3.536700e+03 s, 00000229491448 kB] :: time step 0095800 :: write_image_to_disk 0
[Rank 0000000, 3.543798e+03 s, 00000229491448 kB] :: time step 0096000 :: write_image_to_disk 0
[Rank 0000000, 3.551036e+03 s, 00000230529036 kB] :: time step 0096200 :: write_image_to_disk 0
[Rank 0000000, 3.558141e+03 s, 00000230529036 kB] :: time step 0096400 :: write_image_to_disk 0
[Rank 0000000, 3.565238e+03 s, 00000230529036 kB] :: time step 0096600 :: write_image_to_disk 0
[Rank 0000000, 3.572337e+03 s, 00000230529036 kB] :: time step 0096800 :: write_image_to_disk 0
[Rank 0000000, 3.579436e+03 s, 00000230529036 kB] :: time step 0097000 :: write_image_to_disk 0
[Rank 0000000, 3.586671e+03 s, 00000231566624 kB] :: time step 0097200 :: write_image_to_disk 0
[Rank 0000000, 3.593769e+03 s, 00000231566624 kB] :: time step 0097400 :: write_image_to_disk 0
[Rank 0000000, 3.600867e+03 s, 00000231566624 kB] :: time step 0097600 :: write_image_to_disk 0
[Rank 0000000, 3.607966e+03 s, 00000231566624 kB] :: time step 0097800 :: write_image_to_disk 0
[Rank 0000000, 3.615065e+03 s, 00000231566624 kB] :: time step 0098000 :: write_image_to_disk 0
[Rank 0000000, 3.622304e+03 s, 00000232604212 kB] :: time step 0098200 :: write_image_to_disk 0
[Rank 0000000, 3.629400e+03 s, 00000232604212 kB] :: time step 0098400 :: write_image_to_disk 0
[Rank 0000000, 3.636497e+03 s, 00000232604212 kB] :: time step 0098600 :: write_image_to_disk 0
[Rank 0000000, 3.643595e+03 s, 00000232604212 kB] :: time step 0098800 :: write_image_to_disk 0
[Rank 0000000, 3.650694e+03 s, 00000232604212 kB] :: time step 0099000 :: write_image_to_disk 0
[Rank 0000000, 3.657933e+03 s, 00000233641800 kB] :: time step 0099200 :: write_image_to_disk 0
[Rank 0000000, 3.665032e+03 s, 00000233641800 kB] :: time step 0099400 :: write_image_to_disk 0
[Rank 0000000, 3.672130e+03 s, 00000233641800 kB] :: time step 0099600 :: write_image_to_disk 0
[Rank 0000000, 3.679228e+03 s, 00000233641800 kB] :: time step 0099800 :: write_image_to_disk 0
[Rank 0000000, 3.686327e+03 s, 00000233641800 kB] :: time step 0100000 :: write_image_to_disk 1
[Rank 0000000, 3.700495e+03 s, 00000233949544 kB] :: -------------------
[Rank 0000000, 3.700495e+03 s, 00000233949544 kB] :: SIMULATION FINISHED

real	61m47.022s
user	620m22.641s
sys	429m25.838s
