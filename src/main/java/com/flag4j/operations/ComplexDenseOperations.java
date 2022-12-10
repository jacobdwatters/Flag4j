package com.flag4j.operations;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;


/**
 * This class provides methods for computing operations on dense complex tensors.
 */
public class ComplexDenseOperations {

    private ComplexDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param entries1 Entries of first tensor.
     * @param entries2 Entries of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static CNumber[] add(CNumber[] entries1, CNumber[] entries2) {
        ShapeChecks.arrayLengthsCheck(entries1.length, entries2.length);

        CNumber[] sum = new CNumber[entries1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = new CNumber(
                    entries1[i].re + entries2[i].re,
                    entries1[i].im + entries2[i].im
            );
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param entries1 Entries of first tensor.
     * @param entries2 Entries of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static CNumber[] sub(CNumber[] entries1, CNumber[] entries2) {
        ShapeChecks.arrayLengthsCheck(entries1.length, entries2.length);

        CNumber[] diff = new CNumber[entries1.length];

        for(int i=0; i<diff.length; i++) {
            diff[i] = new CNumber(
                    entries1[i].re - entries2[i].re,
                    entries1[i].im - entries2[i].im
            );
        }

        return diff;
    }


    /**
     * Sums all entries in a tensor.
     * @param entries The entries of the tensor.
     * @return The sum of all entries in the tensor.
     */
    public static CNumber sum(CNumber... entries) {
        CNumber sum = new CNumber();

        for(int i=0; i<entries.length; i++) {
            sum.re += entries[i].re;
            sum.im += entries[i].im;
        }

        return sum;
    }


    /**
     * Multiplies all entries in a tensor.
     * @param entries The entries of the tensor.
     * @return The product of all entries in the tensor.
     */
    public static CNumber prod(CNumber... entries) {
        CNumber product;
        double re;
        double im;

        if(entries.length>0) {
            product = entries[0];
            for(int i=1; i<entries.length; i++) {
                re = product.re*entries[i].re - product.im*entries[i].im;
                im = product.re*entries[i].im + product.im*entries[i].re;

                product.re = re;
                product.im = im;
            }
        } else {
            product = new CNumber();
        }

        return product;
    }
}
