package com.flag4j.complex_matrix;

import com.flag4j.CustomAssertions;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import com.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixElemDivTests {
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realTestCase() {
        double[][] bEntries;
        Matrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.0)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {12.3, 4.45, -878.2},
                {3.456, 3.45, -65.44},
                {4.566, 0, 37.45},
                {-1, -0.0000002, 94.3}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber("10.040650406504064-0.7560975609756098i"), new CNumber("10.157303370786517-0.0074831460674157305i"), new CNumber("-0.00614894101571396")},
                {new CNumber("0.28935185185185186"), new CNumber("0.0-215.3913043478261i"), new CNumber("0.527200488997555+1.4211491442542787i")},
                {new CNumber("1668.2971966710468"), new CNumber(Double.NaN, Double.NaN), new CNumber("0.0")},
                {new CNumber("-3.141592653589793-3.141592653589793i"), new CNumber("-46073311.61750001-75500000.0i"), new CNumber("-0.042417815482502653")}};
        exp = new CMatrix(expEntries);

        CustomAssertions.assertEqualsNaN(exp, A.elemDiv(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.elemDiv(finalB));
    }


    @Test
    void complexTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ------------------- Sub-case 1 -------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber(1)},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1519.0500000000002,-114.39000000000001), new CNumber(201.14000000000001, -0.148185), new CNumber(-4742.280000000001)},
                {new CNumber(3.456), new CNumber(0.0, -2563.695), new CNumber(2257.68, 6085.92)},
                {new CNumber(34781.25387), new CNumber(0.0), new CNumber(0.0)},
                {new CNumber(-3.141592653589793, -3.141592653589793), new CNumber(-0.0000018429324647, -0.00000302), new CNumber(-377.2)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber("0.08130081300813005"), new CNumber("0.22471910112359553"), new CNumber("-0.0011386927806877705")},
                {new CNumber("0.28935185185185186"), new CNumber("0.2898550724637681"), new CNumber(-0.01528117359413203)},
                {new CNumber("0.2190100744634253"), new CNumber(Double.NaN, Double.NaN), new CNumber(Double.NaN, Double.NaN)},
                {new CNumber("-1.0"), new CNumber("-5000000.0"), new CNumber("0.010604453870625663")}};
        exp = new CMatrix(expEntries);

        CustomAssertions.assertEqualsNaN(exp, A.elemDiv(B));

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
        assertThrows(LinearAlgebraException.class, ()->A.elemDiv(finalB));
    }
}
