#include <stdint.h>
#include <stdio.h>

#include "snappy_decompress.h"


/**
 * Attempt to read a varint from the input buffer. The format of a varint
 * consists of little-endian series of bytes where the lower 7 bits are data
 * and the upper bit is set if there are more bytes to read. Maximum size
 * of the varint is 5 bytes.
 *
 * @param input: holds input buffer information
 * @param val: read value of the varint
 * @return False if all 5 bytes were read and there is still more data to
 *		   read, True otherwise
 */
__host__ __device__ static inline bool read_varint32(struct host_buffer_context *input, uint32_t *val)
{
	int shift = 0;
	*val = 0;

	for (uint8_t count = 0; count < 5; count++) {
		int8_t c = (int8_t)(*input->curr++);
		*val |= (c & BITMASK(7)) << shift;
		if (!(c & (1 << 7)))
			return true;
		shift += 7;
	}

	return false;
}

/**
 * Read an unsigned integer from the input buffer. Increments
 * the current location in the input buffer.
 *
 * @param input: holds input buffer information
 * @return Unsigned integer read
 */
__host__ __device__ static uint32_t read_uint32(struct host_buffer_context *input)
{
	uint32_t val = 0;
	for (uint8_t i = 0; i < sizeof(uint32_t); i++) {
		val |= (*input->curr++) << (8 * i);
	}

	return val;
}
		
/**
 * Read the size of the long literal tag, which is used for literals with
 * length greater than 60 bytes.
 *
 * @param input: holds input buffer information
 * @param len: length in bytes of the size to read
 * @return 0 if we reached the end of input buffer, size of literal otherwise
 */
__host__ __device__ static inline uint32_t read_long_literal_size(struct host_buffer_context *input, uint32_t len)
{
	if ((input->curr + len) >= (input->buffer + input->length))
		return 0;

	uint32_t size = 0;
	for (uint32_t i = 0; i < len; i++) {
		size |= (*input->curr++ << (i << 3));
	}
	return size;
}

/**
 * Read a 1-byte offset tag and return the offset of the copy that is read.
 *
 * @param tag: tag byte to parse
 * @param input: holds input buffer information
 * @return 0 if we reached the end of input buffer, offset of the copy otherwise
 */
__host__ __device__ static inline uint16_t make_offset_1_byte(uint8_t tag, struct host_buffer_context *input)
{
	if (input->curr >= (input->buffer + input->length))
		return 0;
	return (uint16_t)(*input->curr++) | (uint16_t)(GET_OFFSET_1_BYTE(tag) << 8);
}

/**
 * Read a 2-byte offset tag and return the offset of the copy that is read.
 *
 * @param tag: tag byte to parse
 * @param input: holds input buffer information
 * @return 0 if we reached the end of input buffer, offset of the copy otherwise
 */
__host__ __device__ static inline uint16_t make_offset_2_byte(uint8_t tag, struct host_buffer_context *input)
{
	UNUSED(tag);

	uint16_t total = 0;
	if ((input->curr + sizeof(uint16_t)) > (input->buffer + input->length))
		return 0;
	else {
		total = (*input->curr & 0xFF) | ((*(input->curr + 1) & 0xFF) << 8);
		input->curr += sizeof(uint16_t);
		return total;
	}
}

/**
 * Read a 4-byte offset tag and return the offset of the copy that is read.
 *
 * @param tag: tag byte to parse
 * @param input: holds input buffer information
 * @return 0 if we reached the end of input buffer, offset of the copy otherwise
 */
__host__ __device__ static inline uint32_t make_offset_4_byte(uint8_t tag, struct host_buffer_context *input)
{
	UNUSED(tag);

	uint32_t total = 0;
	if ((input->curr + sizeof(uint32_t)) > (input->buffer + input->length))
		return 0;
	else {
		total = (*input->curr & 0xFF) |
				((*(input->curr + 1) & 0xFF) << 8) |
				((*(input->curr + 2) & 0xFF) << 16) |
			 ((*(input->curr + 3) & 0xFF) << 24);
		input->curr += sizeof(uint32_t);
		return total;
	}
}

/**
 * Copy and append data from the input bufer to the output buffer.
 *
 * @param input: holds input buffer information
 * @param output: holds output buffer information
 * @param len: length of data to copy over
 */
__host__ __device__ static void writer_append_host(struct host_buffer_context *input, struct host_buffer_context *output, uint32_t len)
{
	//printf("Writing %u bytes at 0x%x\n", len, (input->curr - input->buffer));
	while (len &&
		(input->curr < (input->buffer + input->length)) &&
		(output->curr < (output->buffer + output->length)))
	{
		*output->curr = *input->curr;
		input->curr++;
		output->curr++;
		len--;
	}
}

/**
 * Copy and append previously uncompressed data to the output buffer.
 *
 * @param output: holds output buffer information
 * @param copy_length: length of data to copy over
 * @param offset: where to copy from, offset from current output pointer
 * @return False if offset if invalid, True otherwise
 */
__host__ __device__ static bool write_copy_host(struct host_buffer_context *output, uint32_t copy_length, uint32_t offset)
{
	//printf("OFFSET 0x%x\n",offset);
	//printf("Copying %u bytes from offset=0x%lx to 0x%lx\n", copy_length, (output->curr - output->buffer) - offset, output->curr - output->buffer);
	const uint8_t *copy_curr = output->curr;
	copy_curr -= offset;
	if (copy_curr < output->buffer)
	{
		printf("bad offset!\n");
		return false;
	}
	while (copy_length &&
		output->curr < (output->buffer + output->length))
	{
		*output->curr = *copy_curr;
		copy_curr++;
		output->curr++;
		copy_length -= 1;
	}

	return true;
}

__global__ void snappy_decompress_kernel(struct host_buffer_context *input, struct host_buffer_context *output, uint32_t total_blocks, uint32_t dblock_size, uint32_t *input_offsets, uint8_t **input_currents)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	host_buffer_context input_d, output_d;
	input_d.buffer = input->buffer;
	input_d.length = input->length;
	input_d.curr = input_currents[idx];

	output_d.buffer = output->buffer;
	output_d.length = output->length;
	output_d.curr = output->curr + (idx * dblock_size);

	if(idx < total_blocks)
	{
		uint8_t *block_end = input_d.curr + input_offsets[idx];
	
		while (input_d.curr != block_end) {	
			uint16_t length;
			uint32_t offset;
			const uint8_t tag = *input_d.curr++;
			//printf("Got tag byte 0x%x at index 0x%lx\n", tag, input->curr - input->buffer - 1);

			/* There are two types of elements in a Snappy stream: Literals and
			copies (backreferences). Each element starts with a tag byte,
			and the lower two bits of this tag byte signal what type of element
			will follow. */
			switch (GET_ELEMENT_TYPE(tag))
			{
			case EL_TYPE_LITERAL:
				/* For literals up to and including 60 bytes in length, the upper
				 * six bits of the tag byte contain (len-1). The literal follows
				 * immediately thereafter in the bytestream.
				 */
				length = GET_LENGTH_2_BYTE(tag) + 1;
				if (length > 60)
				{
					length = read_long_literal_size(&input_d, length - 60) + 1;
				}

				writer_append_host(&input_d, &output_d, length);
				break;

			/* Copies are references back into previous decompressed data, telling
			 * the decompressor to reuse data it has previously decoded.
			 * They encode two values: The _offset_, saying how many bytes back
			 * from the current position to read, and the _length_, how many bytes
			 * to copy.
			 */
			case EL_TYPE_COPY_1:
				length = GET_LENGTH_1_BYTE(tag) + 4;
				offset = make_offset_1_byte(tag, &input_d);
				if (!write_copy_host(&output_d, length, offset))
					return;
				break;

			case EL_TYPE_COPY_2:
				length = GET_LENGTH_2_BYTE(tag) + 1;
				offset = make_offset_2_byte(tag, &input_d);
				if (!write_copy_host(&output_d, length, offset))
					return;
				break;

			case EL_TYPE_COPY_4:
				length = GET_LENGTH_2_BYTE(tag) + 1;
				offset = make_offset_4_byte(tag, &input_d);
				if (!write_copy_host(&output_d, length, offset))
					return;
				break;
			}
		}
	}
}

snappy_status setup_decompression(struct host_buffer_context *input, struct host_buffer_context *output, struct program_runtime *runtime)
{
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);

	// Read the decompressed length
	uint32_t dlength;
	if (!read_varint32(input, &dlength)) {
		fprintf(stderr, "Failed to read decompressed length\n");
		return SNAPPY_INVALID_INPUT;
	}

	// Check that uncompressed length is within the max we can store
	if (dlength > output->max) {
		fprintf(stderr, "Output length is too big: max=%ld len=%d\n", output->max, dlength);
		return SNAPPY_BUFFER_TOO_SMALL;
	}

	// Allocate output buffer
    //printf("dlength, aligned, aligned | bitmasked %d %d %d\n",dlength, ALIGN(dlength,8), ALIGN(dlength, 8) | BITMASK(11));
	output->buffer = (uint8_t *)malloc(ALIGN(dlength, 8) | BITMASK(11));
	//output->buffer = (uint8_t *)malloc(ALIGN(dlength, 8));
	output->curr = output->buffer;
	output->length = dlength;

	gettimeofday(&end, NULL);
	runtime->pre = get_runtime(&start, &end);

	return SNAPPY_OK;
}


snappy_status setup_decompression_cuda(struct host_buffer_context *input, struct host_buffer_context *output, struct program_runtime *runtime)
{
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);

	// Read the decompressed length
	uint32_t dlength;
	if (!read_varint32(input, &dlength)) {
		fprintf(stderr, "Failed to read decompressed length\n");
		return SNAPPY_INVALID_INPUT;
	}

	// Check that uncompressed length is within the max we can store
	if (dlength > output->max) {
		fprintf(stderr, "Output length is too big: max=%ld len=%d\n", output->max, dlength);
		return SNAPPY_BUFFER_TOO_SMALL;
	}

	// Allocate output buffer
    //printf("dlength, aligned, aligned | bitmasked %d %d %d\n",dlength, ALIGN(dlength,8), ALIGN(dlength, 8) | BITMASK(11));
	//output->buffer = (uint8_t *)malloc(ALIGN(dlength, 8) | BITMASK(11));
	checkCudaErrors(cudaMallocManaged(&output->buffer, ALIGN(dlength, 8) | BITMASK(11)));

	output->curr = output->buffer;
	output->length = dlength;

	gettimeofday(&end, NULL);
	runtime->pre = get_runtime(&start, &end);

	return SNAPPY_OK;
}

snappy_status snappy_decompress_host(struct host_buffer_context *input, struct host_buffer_context *output)
{
	// Read the decompressed block size
	uint32_t dblock_size;
	if (!read_varint32(input, &dblock_size)) {
		fprintf(stderr, "Failed to read decompressed block size\n");
		return SNAPPY_INVALID_INPUT;
	}

	while (input->curr < (input->buffer + input->length)) {
		// Read the compressed block size
		uint32_t compressed_size = read_uint32(input);	
		uint8_t *block_end = input->curr + compressed_size;
	
		while (input->curr != block_end) {	
			uint16_t length;
			uint32_t offset;
			const uint8_t tag = *input->curr++;
			//printf("Got tag byte 0x%x at index 0x%lx\n", tag, input->curr - input->buffer - 1);

			/* There are two types of elements in a Snappy stream: Literals and
			copies (backreferences). Each element starts with a tag byte,
			and the lower two bits of this tag byte signal what type of element
			will follow. */
			switch (GET_ELEMENT_TYPE(tag))
			{
			case EL_TYPE_LITERAL:
				/* For literals up to and including 60 bytes in length, the upper
				 * six bits of the tag byte contain (len-1). The literal follows
				 * immediately thereafter in the bytestream.
				 */
				length = GET_LENGTH_2_BYTE(tag) + 1;
				if (length > 60)
				{
					length = read_long_literal_size(input, length - 60) + 1;
				}

				writer_append_host(input, output, length);
				break;

			/* Copies are references back into previous decompressed data, telling
			 * the decompressor to reuse data it has previously decoded.
			 * They encode two values: The _offset_, saying how many bytes back
			 * from the current position to read, and the _length_, how many bytes
			 * to copy.
			 */
			case EL_TYPE_COPY_1:
				length = GET_LENGTH_1_BYTE(tag) + 4;
				offset = make_offset_1_byte(tag, input);
				if (!write_copy_host(output, length, offset))
					return SNAPPY_INVALID_INPUT;
				break;

			case EL_TYPE_COPY_2:
				length = GET_LENGTH_2_BYTE(tag) + 1;
				offset = make_offset_2_byte(tag, input);
				if (!write_copy_host(output, length, offset))
					return SNAPPY_INVALID_INPUT;
				break;

			case EL_TYPE_COPY_4:
				length = GET_LENGTH_2_BYTE(tag) + 1;
				offset = make_offset_4_byte(tag, input);
				if (!write_copy_host(output, length, offset))
					return SNAPPY_INVALID_INPUT;
				break;
			}
		}
	}

	return SNAPPY_OK;
}


snappy_status snappy_decompress_cuda(struct host_buffer_context *input, struct host_buffer_context *output, struct program_runtime *runtime)
{
	uint32_t total_blocks = 0;

	// Read the decompressed block size
	uint32_t dblock_size;
	if (!read_varint32(input, &dblock_size)) {
		fprintf(stderr, "Failed to read decompressed block size\n");
		return SNAPPY_INVALID_INPUT;
	}

	//total dblock_size (32K) output blocks
	total_blocks = (output->length + dblock_size - 1) / dblock_size;

	//CUDA calculation for grid and threads per block
	dim3 block(1);
	dim3 grid(total_blocks);
	if(runtime->blocks == 0 && runtime->threads_per_block == 0) //only set blocks and threads_per_block if user didn't set them
	{
		if (total_blocks >= 1024 * 1024 * 1024)
		{
			block.x = 512;
			grid.x = (unsigned int) ceil(total_blocks * 1.0 / block.x);
		}
	}
	else
	{
		grid.x = runtime->blocks;
		block.x = runtime->threads_per_block;
	}
	

	printf("---\nTotal blocks = %d\n", total_blocks);
	printf("grid.x = %d , block.x = %d\n---\n", grid.x, block.x);

	//calculate int input offset for each GPU thread. Since compressed blocks blocks are not distanced equally
	//we have to get the starting location of each block.
	uint8_t **input_currents;
	uint32_t *input_offsets;
	checkCudaErrors(cudaMallocManaged(&input_currents, sizeof(uint8_t *) * total_blocks));
	checkCudaErrors(cudaMallocManaged(&input_offsets, sizeof(uint32_t) * total_blocks));

	int i = 0;
	while (input->curr < (input->buffer + input->length)) {
		// Read the compressed block size
		uint32_t compressed_size = read_uint32(input);
		input_currents[i] = input->curr;
		input_offsets[i] = compressed_size;	
		input->curr += compressed_size;
		i++;
		//printf("block %d compressred size = %d\n",i, compressed_size);
	}

	int device = -1;
  	cudaGetDevice(&device);
	cudaMemPrefetchAsync(input_currents,sizeof(uint8_t *) * total_blocks , device, NULL);
	cudaMemPrefetchAsync(input_offsets,sizeof(uint32_t) * total_blocks , device, NULL);
  	cudaMemPrefetchAsync(output->buffer, output->total_size, device, NULL);
	cudaMemPrefetchAsync(input->buffer, input->total_size, device, NULL);


	snappy_decompress_kernel<<<grid,block,0>>>(input, output, total_blocks, dblock_size, input_offsets, input_currents);
	checkCudaErrors(cudaDeviceSynchronize());

	return SNAPPY_OK;
}	
