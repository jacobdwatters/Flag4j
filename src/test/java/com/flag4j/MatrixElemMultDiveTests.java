package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixElemMultDiveTests {

    Matrix A, B, exp, act;
    CMatrix BComplex, expComplex, actComplex;
    double[][] aEntries, bEntries, expEntries;
    CNumber[][] bCEntries, expCEntries;

    @Test
    void elemMultTest() {
        // ------------- Sub-case 1 ----------------
        aEntries = new double[][]{{1, 2.34, 0.987243}, {-83.331, 33.4, 8.973}};
        bEntries = new double[][]{{8.43, -13.234, 111.44}, {-3.44, 8.23, 0.00001231}};
        expEntries = new double[][]{{8.43, 2.34*-13.234, 0.987243*111.44}, {-83.331*-3.44, 33.4*8.23, 8.973*0.00001231}};

        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        act = A.elemMult(B);

        assertArrayEquals(exp.entries, act.entries);

        // ------------- Sub-case 2 ----------------
        aEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}, {Double.POSITIVE_INFINITY, 8.24}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}, {Double.NEGATIVE_INFINITY, Double.NaN}};
        expEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}, {Double.NEGATIVE_INFINITY, Double.NaN}};

        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        act = A.elemMult(B);

        assertArrayEquals(exp.entries, act.entries);
    }


    @Test
    void elemMultComplexTest() {
        // ------------- Sub-case 1 ----------------
        aEntries = new double[][]{{1, 2.34, 0.987243}, {-83.331, 33.4, -8.973}};
        bCEntries = new CNumber[][]{{new CNumber("0+1.56i"), new CNumber("2.345"), new CNumber("123.3 + 1.2i")},
                {new CNumber("-34-9i"), new CNumber("45.3+3.13i"), new CNumber("-14.2")}};
        expCEntries = new CNumber[][]{{new CNumber("0+1.56i"), new CNumber("5.4873"), new CNumber("121.7270619 + 1.1846915999999998i")},
                {new CNumber("2833.254+749.979i"), new CNumber("1513.0199999999998 + 104.54199999999999i"), new CNumber("127.4166")}};

        A = new Matrix(aEntries);
        BComplex = new CMatrix(bCEntries);
        expComplex = new CMatrix(expCEntries);

        actComplex = A.elemMult(BComplex);

        assertArrayEquals(expComplex.entries, actComplex.entries);
    }


    @Test
    void elemDivTest() {
        // ------------- Sub-case 1 ----------------
        aEntries = new double[][]{{1, 2.34, 0.987243}, {-83.331, 33.4, 8.973}};
        bEntries = new double[][]{{8.43, -13.234, 111.44}, {-3.44, 8.23, 0.00001231}};
        expEntries = new double[][]{{1.0/8.43, 2.34/-13.234, 0.987243/111.44}, {-83.331/-3.44, 33.4/8.23, 8.973/0.00001231}};

        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        act = A.elemDiv(B);

        assertArrayEquals(exp.entries, act.entries);

        // ------------- Sub-case 2 ----------------
        aEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}, {Double.POSITIVE_INFINITY, 8.24}};
        bEntries = new double[][]{{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}, {Double.NEGATIVE_INFINITY, Double.NaN}};
        expEntries = new double[][]{{Double.NaN, Double.NaN}, {Double.NaN, Double.NaN}};

        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        act = A.elemDiv(B);

        assertArrayEquals(exp.entries, act.entries);
    }


    @Test
    void elemDivComplexTest() {
        // ------------- Sub-case 1 ----------------
        aEntries = new double[][]{{1, 2.34, 0.987243}, {-83.331, 33.4, -8.973}};
        bCEntries = new CNumber[][]{{new CNumber("0+1.56i"), new CNumber("2.345"), new CNumber("123.3 + 1.2i")},
                {new CNumber("-34-9i"), new CNumber("45.3+3.13i"), new CNumber("-14.2")}};

        expCEntries = new CNumber[][]{{new CNumber("-0.641025641025641i"), new CNumber("0.997867803837953"), new CNumber(0.008006078656540605,-7.791804045294992E-5)},
                {new CNumber("2.2904236054971707-0.6062886014551334i"), new CNumber("0.7338035854439933 - 0.05070210204061146i"), new CNumber("0.6319014084507043")}};

        A = new Matrix(aEntries);
        BComplex = new CMatrix(bCEntries);
        expComplex = new CMatrix(expCEntries);

        actComplex = A.elemDiv(BComplex);

        assertArrayEquals(expComplex.entries, actComplex.entries);
    }
}
