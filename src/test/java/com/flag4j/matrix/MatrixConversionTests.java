package com.flag4j.matrix;


import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

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
        CVector exp;

        // --------------------- Sub-case 1  ---------------------
        aEntries = new double[][]{{1, 2, 3, 0.0000245}, {452.745, -8234, -2.234, 345.324}};
        A = new Matrix(aEntries);
        exp = new CVector(ArrayUtils.flatten(aEntries));

        assertEquals(exp, A.toVector());
    }
}
