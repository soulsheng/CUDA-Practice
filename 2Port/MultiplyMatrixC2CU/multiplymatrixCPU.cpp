
int  multiplymatrixCPU( float* a, float* b, float* c, int n )
{
	for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			for (int k=0;k<n;k++)
			{
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
	return 0;
}

int  multiplymatrixCPU2( float* a, float* b, float* c, int n )
{
	for (int ii=0;ii<n*n;ii++)
	{
		int i = ii/n;
		int j = ii%n;
		for (int k=0;k<n;k++)
		{
			c[i*n+j] += a[i*n+k] * b[k*n+j];
		}
	}
	return 0;
}

int multiplymatrixCPU3( float* a, float* b, float* c, int n )
{

	for (int i=0;i<n/2;i++)
	{
		for (int j=0;j<n/2;j++)
		{
			int cOffset = 2*n*i + 2*j;
			for (int k=0;k<n/2;k++)
			{
				int aOffset = 2*n*i + 2*k;
				int bOffset = 2*n*k + 2*j;
				
				float cBlockOne[2][2];
				for(int l=0;l<2;l++)
					for(int m=0;m<2;m++)
						cBlockOne[l][m] = 0.0f;

				for(int l=0;l<2;l++)
					for(int m=0;m<2;m++)
						for (int p=0;p<2;p++)
						{
							cBlockOne[l][m] += a[aOffset+l*n+p] * b[bOffset+p*n+m];
						}

				for(int l=0;l<2;l++)
					for(int m=0;m<2;m++)
						c[cOffset + l*n +m ] += cBlockOne[l][m];
			}
		}
	}

	return 0;
}
