Interpreting SCAMP Output
=========================

Interpreting the Matrix Profile
*******************************

When using z-normalized euclidean distance (the default). Peaks in the matrix profile represent time series discords (anomalies) and valleys represent motifs in the data.

When using Pearson Correlation (CLI flag: --output_pearson, pyscamp argument: pearson=true), then the opposite is true; Peaks in the matrix profile correspond to motifs and valleys correspond to discords.

Please see the Matrix Profile Tutorial (slides available `here <https://www.cs.ucr.edu/~eamonn/Matrix_Profile_Tutorial_Part1.pdf>`_ for more information on how to interpret and use the matrix profile.

For more research background on the matrix profile you can find links to the research on the `Matrix Profile Homepage <https://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_.

Missing Data
************

Missing data is specified using NaNs. Any subsequence containing NaNs will not match any other subsequence. The nearest neighbor of a subsequence containing NaN is defined as NaN. If there is no match found for a particular subsequence, it's nearest neighbor index will be set to -1.

Flat Regions
************

Some datasets contain flat regions. In SCAMP we consider a flat region any region of the time series input that remains constant for at least one whole subsequence. Constant regions have a mean-centered norm of 0, which means that their Pearson Correlation and z-normalized euclidean distance are **undefined**. Totally flat regions will not match any subsequence and will output NaN as their nearest neighbor distance.

There can also be regions that are **almost flat**, these regions contain values that are on average very close to the mean value of the subsequences and so have a mean-centered norm which is very small. These sequences can be problematic as floating point roundoff errors can cause problems with SCAMP's output. Internally SCAMP produces the Pearson Correlation by dividing intermediate output by these precomputed norms. If a norm is extremely small, this can cause large floating point roundoff errors. If you are using Pearson Correlation and you encounter a nearest neighbor with correlation greater than 1.0, this is likely the cause for that.

We have tried to prevent catastrophic roundoff error in SCAMP by using an epsilon (currently hardcoded to 1e-13). If the sum of squared error from the mean for the values in a particular subsequence is less than this epsilon, then SCAMP considers the subsequence totally flat (as defined above). This will cause NaN to be output for that subsequence.

In addition, by default we use compensated arithemtic to compute the subsequence means and a high-precision brute force O(nm) approcach for computing the subsequence norms. With this approach precomputation can be expensive when a very large subsequence length is chosen in combonation with a large input size. If you would like to speed up the precomputation with a potential loss of accuracy you can specify (CLI flag: --high_precision_precompute=false) as an option to use a faster, more imprecise method.

SCAMP Output Precision
**********************

While distance computation is done in the recommended 64 bit precision by default. Most of SCAMP's profile types output with 32 bit precision. Intermediate output is compared with 32-bit precision for many of the profile types, therefore SCAMP can only output in 32-bit precision.

Due to the above any nearest neighbors correlations/distances that are indistinguishable in 32 bit precision could be output as the nearest neighbor in SCAMP.
