
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <tchar.h>

#include "gpujpeg.h"

int main(int argc, char **argv)
{
	struct gpujpeg_parameters param;
	gpujpeg_set_default_parameters(&param);  

	struct gpujpeg_image_parameters param_image;

	param_image.width = 1920;
	param_image.height = 1080;
	param_image.comp_count = 3;
	// (for now, it must be 3)
	param_image.color_space = GPUJPEG_RGB; 
	// or GPUJPEG_YCBCR_ITU_R or GPUJPEG_YCBCR_JPEG
	// (default value is GPUJPEG_RGB)
	param_image.sampling_factor = GPUJPEG_4_4_4;

	if (gpujpeg_init_device(0, GPUJPEG_VERBOSE))
	{
		printf("Unable to init device\n");
		return -1;
	}

	struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(&param, &param_image);
	
	if (encoder == NULL)
		return -1;

	int image_size = 0;
	uint8_t* image = NULL;
	
	//if (gpujpeg_image_load_from_file("input_image.rgb", &image, &image_size) != 0)
	//	return -1;

	/*
	uint8_t* encoder_input = NULL;
	uint8_t* image_compressed = NULL;
	int image_compressed_size = 0;
	if (gpujpeg_encoder_encode(encoder, &encoder_input, &image_compressed, &image_compressed_size) != 0 )
		return -1;
	*/
	return 0;
}
