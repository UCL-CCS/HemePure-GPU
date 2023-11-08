#include <cmath>
#include <list>
#include <map>
#include <algorithm>
#include <zlib.h>

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <mpi.h>
#include <omp.h>

#include "geometry/Geometry.h"
#include "io/writers/xdr/XdrMemReader.h"
	


constexpr uint32_t HemeLbMagicNumber = 0x686c6221;
constexpr uint32_t GmyMagicNumber = 0x676d7904;
constexpr uint32_t GmyNativeMagicNumber = 0x676d7905;
constexpr uint32_t GmyVersionNumber = 4;
constexpr uint32_t GmyNativeVersionNumber = 1;
constexpr size_t PreambleBytes = 32;

#ifdef HEMELB_USE_GMYPLUS
constexpr size_t HeaderRecordSize = 16;
#else
constexpr size_t HeaderRecordSize = 12;
#endif

struct NonEmptyHeaderRecord {
  uint64_t   blockNumber;  // 8 bytes
  uint64_t   fileOffset;   // 8 bytes
  uint32_t sites;          // 4 bytes
  uint32_t bytes;           // 4 bytes
  uint32_t uncompressedBytes; // 4 bytes  Total 28 bytes
  uint32_t weights;			  // 4 bytes  Total 32 bytes
};


struct PreambleInfo {
  uint32_t HemeLBMagic;
  uint32_t GmyMagic;
  uint32_t Version;
  uint32_t BlocksX;
  uint32_t BlocksY;
  uint32_t BlocksZ;
  uint32_t BlockSize;
  uint32_t Padding;
  PreambleInfo() : HemeLBMagic(HemeLbMagicNumber),
				   GmyMagic(GmyMagicNumber),
				   Version(GmyVersionNumber),
			       BlocksX(0), BlocksY(0), BlocksZ(0),
				   BlockSize(0), Padding(0) {}
};

struct OutputPreambleInfo {
  uint32_t HemeLBMagic;
  uint32_t GmyNativeMagic;

  uint32_t Version;
  uint32_t BlocksX;

  uint32_t BlocksY;
  uint32_t BlocksZ;

  uint32_t BlockSize;
  uint32_t MaxCompressedBytes;

  uint32_t MaxUncompressedBytes;
  uint32_t HeaderOffset;

  uint64_t NonEmptyBlocks;
  uint64_t DataOffset;
  OutputPreambleInfo() : HemeLBMagic(HemeLbMagicNumber),
					     GmyNativeMagic(GmyNativeMagicNumber),
						 Version(GmyNativeVersionNumber),
					     BlocksX(0), BlocksY(0), BlocksZ(0),
					     MaxCompressedBytes(0), MaxUncompressedBytes(0),
					     HeaderOffset(56),
					     NonEmptyBlocks(0),
					     DataOffset(0) {}

};
size_t ReadPreamble(PreambleInfo& pinfo, FILE *fp)
{
	std::vector<char> preambleBuffer(PreambleBytes);

	fseek(fp,0,SEEK_SET);
	size_t bytesRead = fread((void *)&preambleBuffer[0], sizeof(unsigned char), PreambleBytes, fp);
	if( bytesRead != PreambleBytes ) {
	  std::cerr << "Incorrect bytes read in preamble\n";
	  abort();
    }

	// Create an Xdr translator based on the read-in data.
	hemelb::io::writers::xdr::XdrReader preambleReader = hemelb::io::writers::xdr::XdrMemReader(&preambleBuffer[0],
					PreambleBytes);

// Read in housekeeping values.
	if( !preambleReader.readUnsignedInt(pinfo.HemeLBMagic) ) { std::cerr << "HemeLBMagic Read returned failure\n"; abort();} 
	if( !preambleReader.readUnsignedInt(pinfo.GmyMagic) ) { std::cerr << "GeometryMagic Read returned failure\n"; abort(); }  
	if( !preambleReader.readUnsignedInt(pinfo.Version) ) { std::cerr << "Version Read returned failure\n"; abort(); }

	// Check the value of the HemeLB magic number.
	if (pinfo.HemeLBMagic != HemeLbMagicNumber) {
	  std::cerr <<  "This file does not start with the HemeLB magic number."
					<< " Expected: " << HemeLbMagicNumber
					<< " Actual: " << pinfo.HemeLBMagic;
	  abort();	
	}

	// Check the value of the geometry file magic number.
	if ( pinfo.GmyMagic != GmyMagicNumber) {
	  std::cerr << "This file does not have the geometry magic number."
					<< " Expected: " << unsigned(GmyMagicNumber)
					<< " Actual: " << pinfo.GmyMagic;
      abort();
	}

    if (pinfo.Version != GmyVersionNumber) {
	   std::cerr << "Version number incorrect."
					<< " Supported: " << unsigned(GmyVersionNumber)
					<< " Input: " << pinfo.Version;
	   abort();
	}

    // Read in the values.
	if(!preambleReader.readUnsignedInt(pinfo.BlocksX) ) { std::cerr << "BlocksX Read returned failure\n"; abort();} 
	if(!preambleReader.readUnsignedInt(pinfo.BlocksY) ) { std::cerr << "BlocksY Read returned failure\n"; abort();} 
	if(!preambleReader.readUnsignedInt(pinfo.BlocksZ) ) { std::cerr << "BlocksZ Read returned failure\n"; abort(); }
	if(!preambleReader.readUnsignedInt(pinfo.BlockSize) ){ std::cerr << "BlockSize Read returned failure\n";abort(); }
	if(!preambleReader.readUnsignedInt(pinfo.Padding) ) { std::cerr << "Padding Reader returned failure\n"; abort(); }

	std::cout << "Preamble Info:  sizeof of preamble: " << sizeof(PreambleInfo) << "\n";
	std::cout << "Blocks volume:  BX = " << pinfo.BlocksX << " BY = " << pinfo.BlocksY << " BZ = " << pinfo.BlocksZ << "\n";
	std::cout << "Block size:  " << pinfo.BlockSize <<"\n";
	size_t ret_val= (size_t)pinfo.BlocksX * (size_t)pinfo.BlocksY * (size_t)pinfo.BlocksZ;
    std::cout << "Total Number of Blocks (including empty) : " << ret_val << "\n";
	return ret_val;
}

std::unordered_map<size_t, NonEmptyHeaderRecord> headerInfo;


size_t ReadHeader(size_t numBlocks, FILE* fp)
{
	size_t fileOffset = 0; // This is where the data starts in the input filei, the offset is from the end of the header.
	constexpr size_t maxBufLength = 128*1024;
    constexpr size_t maxBufBytes  = HeaderRecordSize*maxBufLength;

    char dataBuffer[maxBufBytes];

	size_t nChunks = numBlocks/maxBufLength;
	if ( numBlocks % maxBufLength != 0 ) nChunks++; // Mop up chunk

	std::cerr << "Reading Header: " << nChunks << " chunks of up to " << maxBufLength << " each\n";

	int nChunkBy10 = nChunks/10;

	size_t nonEmptyBlocks = 0;
	for(size_t chunk = 0; chunk < nChunks; ++chunk) {
		if ( nChunkBy10 > 0 ) {
		  if ( chunk % nChunkBy10 == 0 ) std::cout << "." << std::flush;
        }
		size_t first = chunk*maxBufLength;
		size_t last  = (chunk+1)*maxBufLength;
		if( last > numBlocks ) last = numBlocks;
		size_t nElem = last-first;

		// Read
		size_t bytesRead = fread((void *)dataBuffer, HeaderRecordSize, nElem, fp);
		if (bytesRead != nElem ) {
		  std::cerr << "Could Not Read Header Data  expected: " << nElem << " bytes, but got " << bytesRead << " bytes\n";
		  abort();
	    }

	    // Now process the data
		hemelb::io::writers::xdr::XdrMemReader headerReader(&dataBuffer[0], HeaderRecordSize*nElem);

		for(size_t  block = first; block < last; block++) {
		  uint32_t sites, bytes, uncompressedBytes;
		  uint32_t weights = 0;

		  if( ! headerReader.readUnsignedInt(sites) ) { std::cerr << "Could not read sites\n"; abort(); }
#ifdef HEMELB_USE_GMYPLUS
		  // overwrite weights
		  if( ! headerReader.readUnsignedInt(weights) ) {  std::cerr << "Could not read weights\n"; abort(); }
#endif
		  if( ! headerReader.readUnsignedInt(bytes) )  { std::cerr << "Could not read bytess\n"; abort(); }
		  if( !headerReader.readUnsignedInt(uncompressedBytes) ) { std::cerr << "Could  not read uncompressed bytes\n"; abort(); }

		  if( sites > 0 ) {
	 	    headerInfo[block] = { block, fileOffset, sites, bytes, uncompressedBytes, weights };
	 	    nonEmptyBlocks++;  // Increase the non empty block count
			fileOffset += bytes;  // Increase the offset pointer
		
		  }// sites > 0 

		} // for blocks in chunk

     }	// over chunks
	 std::cout << "\n";

	 return nonEmptyBlocks;
}

constexpr size_t max_block_buffer_length = 8*8*8 *( sizeof(uint32_t)  // Is Fluid
											+ 27*( sizeof(uint32_t) // MaxNumVector * ( Intersection Type
												  + sizeof(float)+sizeof(uint32_t) // distance, ioletID
												 ) 
											+ sizeof(uint32_t) // WallNormal
											 + 3*sizeof(float)); // Normal Dirs

static char uncompressed_block_buffer[ max_block_buffer_length ];
static char compressed_buffer[ max_block_buffer_length ];

int main(int argc, char *argv[])
{
	FILE *fp = nullptr;
	if ( argc < 3 ) {
		printf("Usage: io_tool <filename> <outfile>\n");
		exit(1);
	}

	std::string filename(argv[1]);
	std::cout << "About to process " << filename << "\n";

	fp = fopen(filename.c_str(), "r");
    PreambleInfo preambleInput;

	double preambleStart = omp_get_wtime();
	size_t numBlocks = ReadPreamble(preambleInput, fp);
	double preambleEnd = omp_get_wtime();
	std::cout << "Preamble read time: " << preambleEnd - preambleStart << " sec.\n";

 
	OutputPreambleInfo outInfo;
	outInfo.BlocksX = preambleInput.BlocksX;
	outInfo.BlocksY = preambleInput.BlocksY;
	outInfo.BlocksZ = preambleInput.BlocksZ;
	outInfo.BlockSize = preambleInput.BlockSize;
	std::cout << "Size of Header = " << (double)(numBlocks * HeaderRecordSize) / (double)(1024*1024) << " MiB\n";
	double headerReadStart = omp_get_wtime();
	outInfo.NonEmptyBlocks = (uint64_t)ReadHeader(numBlocks, fp);
	double headerReadEnd = omp_get_wtime();

	std::cout << "Read " << outInfo.NonEmptyBlocks << " non Empty Blocks. Ratio of nonEmpty to Empty Blocks is " << (double)outInfo.NonEmptyBlocks/(double)numBlocks << "\n";
	std::cout << "Header Read and Process Time: " << headerReadEnd - headerReadStart << " sec.\n";

	// Some statistics. Compare Data size and Uncompressed Data size
	double statsLoopBegin=omp_get_wtime();

	size_t totalCompressed = 0;
	size_t totalUncompressed = 0;
	size_t maxUncompressed = 0;
	size_t maxCompressed = 0;

	for( auto elem : headerInfo ) {
		totalCompressed += (size_t)elem.second.bytes;
	    totalUncompressed += (size_t)elem.second.uncompressedBytes;
		if ( (size_t)elem.second.uncompressedBytes > maxUncompressed ) maxUncompressed = (size_t)elem.second.uncompressedBytes; 
		if ( (size_t)elem.second.bytes > maxCompressed ) maxCompressed = (size_t)elem.second.bytes; 
	}

	outInfo.MaxCompressedBytes = maxCompressed;
	outInfo.MaxUncompressedBytes = maxUncompressed;
	outInfo.DataOffset = outInfo.HeaderOffset + outInfo.NonEmptyBlocks*sizeof(NonEmptyHeaderRecord);

	double statsLoopEnd=omp_get_wtime();
	std::cout << "Stats Loop: " << statsLoopEnd-statsLoopBegin << " sec.\n";
	std::cout << "Total Compressed Data: " << (double)totalCompressed / (double)(1024*1024) << " MiB\n";
	std::cout << "Total Uncompressed Data: " << (double)totalUncompressed / (double) (1024*1024) << " MiB\n";
	std::cout << "Compression Ratio: " << (double)totalCompressed/(double)totalUncompressed << " \n";
	std::cout << "Max Uncompressed:  " << (double)maxUncompressed /(double)(1024) << " kiB\n";
	std::cout << "Max Compressed: " << (double)maxCompressed /(double)(1024.0) << " kiB\n";
	
	size_t outputSize=sizeof(OutputPreambleInfo)+outInfo.NonEmptyBlocks*sizeof(NonEmptyHeaderRecord)+totalCompressed;

	std::cout << "Expected Output Size: " << outputSize << " bytes ~ " << (double)outputSize/(double)(1024*1024*1024) << " GiB\n";

	if( outInfo.HeaderOffset != sizeof(OutputPreambleInfo) ) { 
	  std::cerr << "OutputPreambleInfo size is not the same as HeaderOffset\n";
	  abort();
    }

    // Write the output header
	
	std::string outfile(argv[2]);
	FILE *op = fopen(outfile.c_str(), "w+");

    // Write New Preamble
	{
		double start = omp_get_wtime();
		// Write new preamble
		size_t objectsWritten = fwrite( &outInfo, sizeof(OutputPreambleInfo), 1, op);
		if ( objectsWritten != 1 ) { 
			std::cerr << "Unable to Write Preamble\n";
 			abort();
		}
		double end = omp_get_wtime();
		std::cout << "New Preamble Write took " << end -start << " sec.\n";
	}

    // Write New Header    
	{
	    double start = omp_get_wtime();
		constexpr size_t elemPerWrite=4096;
		char outbuf[elemPerWrite*sizeof(NonEmptyHeaderRecord)] __attribute__((aligned(4096))); // Page aligned
		NonEmptyHeaderRecord *bufptr =(NonEmptyHeaderRecord *)&outbuf[0];

		auto mapIterator = headerInfo.begin();
		size_t totalWritten = 0;
		while( mapIterator != headerInfo.end() ) {
		  size_t bufidx=0;
		  while(bufidx < elemPerWrite && mapIterator != headerInfo.end()) { 
			 bufptr[bufidx] = (*mapIterator).second;
			 bufidx++;
			 mapIterator++; 
		  }
		  // Now either bufidx = elemPerWrite or
		  // buf idx is the number of items in the buffer, and mapIterator is at the end
		  // In both cases we should now write bufidx items
		  size_t objectsThisWrite = fwrite( (void *)bufptr, sizeof(NonEmptyHeaderRecord), bufidx, op);
		  if ( objectsThisWrite != bufidx ) {
			   std::cout << "Only wrote " << objectsThisWrite << " objects instead of " << bufidx << "\n";
		  }
		  totalWritten +=bufidx;
		}
		double end = omp_get_wtime(); 
	    std::cout << "Wrote " << totalWritten << " sparse headerRecords in " << end -start << " sec. \n"	;
    }

	// Loop to copy the compressed data here
	std::cout << "About to copy rest of data \n";
	{ 
		size_t start = omp_get_wtime();
 
		constexpr size_t dataBufSize = 128*1024;

		unsigned char databuf[dataBufSize]; // 128K Buffer
		size_t dataWritten = 0;
		while( dataWritten <  totalCompressed ) {
		   size_t toWrite = totalCompressed - dataWritten;
		   if ( toWrite > dataBufSize ) toWrite = dataBufSize;
		  
		   // Read into the buffer from fp
		   size_t bytesRead = fread( (void *)&databuf[0], 1, toWrite, fp);
		   if( bytesRead != toWrite ) { 
				std::cerr << "Error Reading Data Segment\n";
				abort();
		   }
		  size_t bytesWritten = fwrite((void *)&databuf[0],1,toWrite, op);
		  if( bytesWritten != toWrite ) { 
				std::cerr << "Error Writing Data Segment\n";
				abort();
		   }
		   dataWritten += toWrite;
		}
		double end = omp_get_wtime();
		std::cout << "Wrote " << (double)totalCompressed / (double)(1024*1024*1024) << " GiB in " << end -start << " sec. "
				  << "Bandwidth = " << ((double)totalCompressed/(double)(1024*1024*1024)) / (end - start) << " GiB/sec.\n";
    }
    fclose(op);
    fclose(fp);

}
