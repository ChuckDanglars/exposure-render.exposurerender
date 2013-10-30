
#include "gpujpeg.h"

#include <QDebug>

QGpuJpegEncoder::QGpuJpegEncoder(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Encoder(0),
	Initialized(false),
	CompressedImage(0),
	CompressedSize(0)
{
}

QGpuJpegEncoder::~QGpuJpegEncoder()
{
	if (this->CompressedImage)
		gpujpeg_image_destroy(this->CompressedImage);

	if (this->Encoder)
		gpujpeg_encoder_destroy(this->Encoder);
}

void QGpuJpegEncoder::Initialize(const unsigned int& Width, const unsigned int& Height, const unsigned int& NoComponentsPerPixel)
{
	qDebug() << "Initializing gpu jpeg encoder";

	if (gpujpeg_init_device(0, 0))
	{
		qDebug() << "Unable to initialize gpu device for encoding";
		return;
	}

	struct gpujpeg_parameters Params;
	
	gpujpeg_set_default_parameters(&Params);

	Params.quality				= 95;
	Params.restart_interval		= 16; 
	Params.interleaved			= 1;

	struct gpujpeg_image_parameters ImageParams;

    gpujpeg_image_set_default_parameters(&ImageParams);

	ImageParams.width			= Width;
    ImageParams.height			= Height;
	ImageParams.comp_count		= NoComponentsPerPixel;
	ImageParams.color_space		= GPUJPEG_RGB; 
    ImageParams.sampling_factor = GPUJPEG_4_4_4;

	gpujpeg_parameters_chroma_subsampling(&Params);

	this->Encoder = gpujpeg_encoder_create(&Params, &ImageParams);
	
	if (this->Encoder == 0)
	{
		qDebug() << "Unable to create encoder";
		return;
	}

	qDebug() << "Gpu jpeg encoder initialized";

	this->Initialized = true;
}

void QGpuJpegEncoder::Encode(unsigned char* Image)
{
	if (!this->Initialized)
	{
		qDebug() << "Unable to encode, encoder not initialized properly";
		return;
	}

	int image_size = this->Encoder->coder.data_size;

	struct gpujpeg_encoder_input EncoderInput;

	gpujpeg_encoder_input_set_image(&EncoderInput, (uint8_t*)Image);
    
	if (gpujpeg_encoder_encode(this->Encoder, &EncoderInput, &this->CompressedImage, &this->CompressedSize) != 0)
	{
		qDebug() << "Unable to encode";
		return;
	}
}

unsigned char* QGpuJpegEncoder::GetCompressedImage(int& Size)
{
	Size = this->CompressedSize;
	return this->CompressedImage;
}

QGpuJpegDecoder::QGpuJpegDecoder(QObject* Parent /*= 0*/) :
	QObject(Parent),
	Decoder(0),
	DecoderOutput(),
	Initialized(false),
	Image(0),
	ImageSize(0)
{
	this->Initialize();
}

QGpuJpegDecoder::~QGpuJpegDecoder()
{
	gpujpeg_image_destroy(this->Image);
	gpujpeg_decoder_destroy(this->Decoder);
}

void QGpuJpegDecoder::Initialize()
{
	qDebug() << "Initializing gpu jpeg decoder";

	if (gpujpeg_init_device(0, 0))
	{
		qDebug() << "Unable to initialize gpu device for decoding";
		return;
	}

	this->Decoder = gpujpeg_decoder_create();
	
	if (this->Decoder == 0)
	{
		qDebug() << "Unable to create decode";
		return;
	}

	this->Initialized = true;
}

void QGpuJpegDecoder::Decode(unsigned char* CompressedImage, const int& CompressedImageSize, int& Width, int& Height, int& NoBytes)
{
	if (!this->Initialized)
	{
		qDebug() << "Unable to decode, decoder not initialized properly";
		return;
	}

	if (Decoder == NULL)
          return;

	gpujpeg_decoder_output_set_default(&this->DecoderOutput);

	if (gpujpeg_decoder_decode(Decoder, CompressedImage, CompressedImageSize, &this->DecoderOutput) != 0 )
	{
		qDebug() << "Unable to decode";
		return;
	}

	this->Image		= this->DecoderOutput.data;
	this->ImageSize	= this->DecoderOutput.data_size;
	
	Width	= this->Decoder->coder.data_width;
	Height	= this->Decoder->coder.data_height;
	NoBytes	= this->DecoderOutput.data_size;
}

unsigned char* QGpuJpegDecoder::GetImage(int& Size)
{
	Size = this->ImageSize;
	return this->Image;
}
