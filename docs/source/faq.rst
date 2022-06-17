Frequently Asked Questions
==========================

This section contains solutions to common troubleshooting issues and questions about the SCAMP repository.

Missing libcufft.so on Linux
****************************

If when trying to run the SCAMP CLI or import pyscamp, you recieve an error message that looks something like: 

``libcufft.so: cannot open shared object file: No such file or directory``

It is likely that CUDA has not been set up correctly on your system. You need to make sure your LD_LIBRARY_PATH is set correctly. You can put the following into your .bashrc or similar: 

``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64``


CUDA/GPUs won't work
********************

A common issue is that CUDA does not get picked up properly when building SCAMP or pyscamp. There are tips on how to deal with this in the documentation but some common tips are:

- Make sure your NVIDIA drivers are up to date.

- Make sure your :doc:`environment </environment>` is set up properly and wherever CUDA is installed is on your PATH. For example on linux usually this is: /usr/local/cuda/bin

- Set FORCE_CUDA=1 (See CLI and pyscamp docs for more information) to validate that CUDA is being detected during installation.

- See the :doc:`pyscamp </pyscamp/intro>` and :doc:`SCAMP CLI </cli>` documentation for more troubleshooting information.

SCAMP's output looks wrong
**************************

By default, SCAMP uses double precision to compute the matrix profile, for some datasets this may not be sufficient. SCAMP provides an ultra precision option which can help reduce floating point roundoff error. You can specify this option by specifying:

- SCAMP CLI: ``--ultra_precision``

- pyscamp: ``precision='ultra'`` as a kwarg to any pyscamp function

Even this option will not be sufficient for some datasets, but it should work for the vast majority. For more information please see the documentation on :doc:`interpreting output </interpreting_output>`.
