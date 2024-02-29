package org.flag4j.complex_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixElemMultTests {
    Shape sparseShape;
    int[] rowIndices, colIndies;

    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realTestCase() {
        double[][] bEntries;
        Matrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1519.0500000000002,-114.39000000000001), new CNumber(201.14000000000001, -0.148185), new CNumber(-4742.280000000001)},
                {new CNumber(3.456), new CNumber(0.0, -2563.695), new CNumber(2257.68, 6085.92)},
                {new CNumber(34781.25387), new CNumber(0.0), new CNumber(0.0)},
                {new CNumber(-3.141592653589793, -3.141592653589793), new CNumber(-0.0000018429324647, -0.00000302), new CNumber(-377.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void complexTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1519.0500000000002,-114.39000000000001), new CNumber(201.14000000000001, -0.148185), new CNumber(-4742.280000000001)},
                {new CNumber(3.456), new CNumber(0.0, -2563.695), new CNumber(2257.68, 6085.92)},
                {new CNumber(34781.25387), new CNumber(0.0), new CNumber(0.0)},
                {new CNumber(-3.141592653589793, -3.141592653589793), new CNumber(-0.0000018429324647, -0.00000302), new CNumber(-377.2)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber(186538.84800000003,-28254.330000000005), new CNumber(9091.523065439502, -13.395924000000003), new CNumber(-25608.312000000005)},
                {new CNumber(3.456), new CNumber(-1905081.7545000003), new CNumber(488100.6000000001, -419928.48)},
                {new CNumber(264944288.38576216), new CNumber(0.0), new CNumber(0.0)},
                {new CNumber(0.0, -19.739208802178716), new CNumber(2.8619999652773907e-05, -5.565656043394e-05), new CNumber(1508.8)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1519.0500000000002,-114.39000000000001), new CNumber(201.14000000000001, -0.148185)},
                {new CNumber(3.456), new CNumber(0.0, -2563.695)},
                {new CNumber(34781.25387), new CNumber(0.0)},
                {new CNumber(-3.141592653589793, -3.141592653589793), new CNumber(-0.0000018429324647, -0.00000302)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void sparseRealTestCase() {
        double[] bEntries;
        CooMatrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.1)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.34, -994.1, 34.5};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(A.shape);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndies);
        expEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("60.568000000000005-0.04462200000000001i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("-0.0"), new CNumber("0.0")},
                {new CNumber("108.38494654884786+108.38494654884786i"), new CNumber("0.0"), new CNumber("-0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.34, -994.1, 34.5};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(56, 191114);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndies);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }


    @Test
    void sparseComplexTestCase() {
        CNumber[] bEntries;
        CooCMatrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.1)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(345.6, 94.1), new CNumber(-9.4, 34), new CNumber(4.4)};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(A.shape);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndies);
        expEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("15624.253530000002+4241.811519999999i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("-0.0"), new CNumber("0.0")},
                {new CNumber("13.823007675795091+13.823007675795091i"), new CNumber("0.0"), new CNumber("-0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(345.6, 94.1), new CNumber(-9.4, 34), new CNumber(4.4)};
        rowIndices = new int[]{0, 2, 3};
        colIndies = new int[]{1, 1, 0};
        sparseShape = new Shape(56, 191114);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndies);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.elemMult(finalB));
    }
}
