package org.flag4j.matrix;


import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixConversionTests {

    double[][] aEntries;
    MatrixOld A;


    @Test
    void toComplexTestCase() {
        CMatrixOld exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new MatrixOld(aEntries);
        exp = new CMatrixOld(aEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toVectorTestCase() {
        VectorOld exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new MatrixOld(aEntries);
        exp = new VectorOld(ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toVector());
    }


    @Test
    void toTensorTestCase() {
        TensorOld exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new MatrixOld(aEntries);
        exp = new TensorOld(A.shape, ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toTensor());
    }
}
