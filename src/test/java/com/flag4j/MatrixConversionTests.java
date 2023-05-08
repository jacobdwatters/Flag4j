package com.flag4j;


import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixConversionTests {

    double[][] aEntries;
    Matrix A;


    @Test
    void toComplexTest() {
        CMatrix exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new CMatrix(aEntries);

        assertEquals(exp, A.toComplex());
    }


    @Test
    void toVectorTest() {
        Vector exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Vector(ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toVector());
    }


    @Test
    void toTensorTest() {
        Tensor exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new Tensor(A.shape.copy(), ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toTensor());
    }
}
