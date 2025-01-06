/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.linalg.ops.common.real;

/**
 * This class provides low level methods for checking tensor properties. These methods can be applied to
 * either sparse or dense real tensors.
 */
public final class RealProperties {

    private RealProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a tensor only contain positive values. If the tensor is sparse, only the non-zero data are considered.
     * @param entries Entries of the tensor in question.
     * @return {@code true} if the tensor contains only positive values; {@code false} otherwise.
     */
    public static boolean isPos(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value<=0) {
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if a tensor only contain negative values. If the tensor is sparse, only the non-zero data are considered.
     * @param entries Entries of the tensor in question.
     * @return {@code true} if the tensor contains only negative values; {@code false} otherwise.
     */
    public static boolean isNeg(double[] entries) {
        boolean result = true;

        for(double value : entries) {
            if(value>=0) {
                result = false;
                break;
            }
        }

        return result;
    }

    /**
     * Checks if an array contains only zeros.
     * @param src Array to check if it only contains zeros.
     * @return True if the {@code src} array contains only zeros.
     */
    public static boolean isZeros(double[] src) {
        boolean allZeros = true;

        for(double value : src) {
            if(value!=0) {
                allZeros = false;
                break;
            }
        }

        return allZeros;
    }


    /**
     * Checks if all data of two arrays are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays have the same length and all data are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays respectively and satisfy
     * {@code |a-b| <= (1E-08 + 1E-05*|b|)}. Otherwise, returns false.
     * @see #allClose(double[], double[], double, double)
     */
    public static boolean allClose(double[] src1, double[] src2) {
        return allClose(src1, src2, 1e-05, 1e-08);
    }


    /**
     * Checks if all data of two arrays are 'close'.
     * @param src1 First array in comparison.
     * @param src2 Second array in comparison.
     * @return True if both arrays have the same length and all data are 'close' element-wise, i.e.
     * elements {@code a} and {@code b} at the same positions in the two arrays respectively and satisfy
     * {@code |a-b| <= (absTol + relTol*|b|)}. Otherwise, returns false.
     * @see #allClose(double[], double[])
     */
    public static boolean allClose(double[] src1, double[] src2, double relTol, double absTol) {
        boolean close = src1.length==src2.length;

        if(close) {
            for(int i=0; i<src1.length; i++) {
                double tol = absTol + relTol*Math.abs(src2[i]);

                if(Math.abs(src1[i]-src2[i]) > tol) {
                    close = false;
                    break;
                }
            }
        }

        return close;
    }


    /**
     * Checks if this tensor only contains ones.
     * @param src Elements of the tensor.
     * @return {@code true} if this tensor only contains ones; {@code false} otherwise.
     */
    public static boolean isOnes(double[] src) {
        boolean allZeros = true;

        for(double value : src) {
            if(value != 1) {
                allZeros = false;
                break; // No need to look further.
            }
        }

        return allZeros;
    }


    /**
     * Checks if <i>any</i> of the elements of a tensor contain a {@link Double#NaN}.
     * @param src Entries of the tensor.
     * @return {@code true} is any entry of {@code src} is {@link Double#NaN}; {@code false} otherwise.
     */
    public static boolean isNaN(double[] src) {
        for(double value : src)
            if(Double.isNaN(value)) return true;

        return false;
    }


    /**
     * Checks if <i>all</i> of the elements of a tensor are finite.
     * @param src Entries of the tensor.
     * @return {@code false} is any entry of {@code src} is not {@link Double#isFinite(double) finite}. Otherwise, returns {@code
     * true}.
     */
    public static boolean isFinite(double[] src) {
        for(double value : src)
            if(!Double.isFinite(value)) return false;

        return true;
    }


    /**
     * Checks if <i>any</i> of the elements of a tensor is infinite.
     * @param src Entries of the tensor.
     * @return {@code true} is any entry of {@code src} is {@link Double#isInfinite(double) infinite}; {@code false} otherwise.
     */
    public static boolean isInfinite(double[] src) {
        for(double value : src)
            if(Double.isInfinite(value)) return true;

        return false;
    }


    /**
     * Computes the minimum value in a tensor. Note, if the data array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The minimum value in the tensor.
     */
    public static double min(double... entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;

        for(double value : entries)
            currMin = Math.min(value, currMin);

        return currMin;
    }


    /**
     * Computes the maximum value in a tensor. Note, if the data array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The maximum value in the tensor.
     */
    public static double max(double... entries) {
        double currMax = (entries.length==0) ? 0 : Double.MIN_NORMAL;

        for(double value : entries)
            currMax = Math.max(value, currMax);

        return currMax;
    }


    /**
     * Computes the minimum absolute value in a tensor. Note, if the data array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The minimum absolute value in the tensor.
     */
    public static double minAbs(double... entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;

        for(double value : entries)
            currMin = Math.min(Math.abs(value), currMin);

        return currMin;
    }


    /**
     * Computes the maximum absolute value in a tensor. Note, if the data array is empty, this method will return 0 allowing
     * this method to be used for real sparse or dense tensors.
     * @param entries Entries of the tensor.
     * @return The maximum absolute value in the tensor.
     */
    public static double maxAbs(double... entries) {
        double currMax = 0;

        for(double value : entries)
            currMax = Math.max(Math.abs(value), currMax);

        return currMax;
    }


    /**
     * Finds the index of the minimum value within a tensor.
     * @param entries The data of the tensor.
     * @return The index of the minimum values within {@code data}. If {@code data.length == 0} then -1 will be returned.
     */
    public static int argmin(double... entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;
        int mindex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            if (entries[i] < currMin) {
                currMin = entries[i];
                mindex = i;
            }
        }

        return mindex;
    }


    /**
     * Finds the index of the maximum value within a tensor.
     * @param entries The data of the tensor.
     * @return The index of the maximum values within {@code data}. If {@code data.length == 0} then -1 will be returned.
     */
    public static int argmax(double... entries) {
        double currMax = (entries.length==0) ? 0 : Double.MIN_NORMAL;
        int maxdex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            if (entries[i] > currMax) {
                currMax = entries[i];
                maxdex = i;
            }
        }

        return maxdex;
    }


    /**
     * Finds the index of the minimum absolute value within a tensor.
     * @param entries The data of the tensor.
     * @return The index of the minimum absolute values within {@code data}. If {@code data.length == 0} then -1 will be
     * returned.
     */
    public static int argminAbs(double... entries) {
        double currMin = (entries.length==0) ? 0 : Double.MAX_VALUE;
        int mindex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            if (Math.abs(entries[i]) < currMin) {
                currMin = entries[i];
                mindex = i;
            }
        }

        return mindex;
    }


    /**
     * Finds the index of the maximum absolute value within a tensor.
     * @param entries The data of the tensor.
     * @return The index of the maximum absolute values within {@code data}. If {@code data.length == 0} then -1 will be
     * returned.
     */
    public static int argmaxAbs(double... entries) {
        double currMax = (entries.length==0) ? 0 : Double.MIN_NORMAL;
        int maxdex = -1;

        for(int i=0, size=entries.length; i<size; i++) {
            if (Math.abs(entries[i]) > currMax) {
                currMax = entries[i];
                maxdex = i;
            }
        }

        return maxdex;
    }
}
