package com.flag4j.matrix;


import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import com.flag4j.dense.Tensor;
import com.flag4j.dense.Vector;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixConversionTests {

    double[][] aEntries;
    Matrix A;


    @Test
    void toComplexTestCase() {
        CMatrix exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new CMatrix(aEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toVectorTestCase() {
        Vector exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Vector(ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toVector());
    }


    @Test
    void toTensorTestCase() {
        Tensor exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Tensor(A.shape.copy(), ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toTensor());
    }
}
