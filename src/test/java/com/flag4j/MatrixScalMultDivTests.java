package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.concurrency.algorithms.addition.ConcurrentAddition;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixScalMultDivTests {
    double factor;
    Matrix A, exp, act;
    double[][] aEntries, expEntries;
    CNumber CFactor;
    CNumber[][] expComplexEntries;
    CMatrix expComplex, actComplex;

    @Test
    void scalMultTest() {
        // ------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        factor = 3.14;
        expEntries = new double[][]{{1*3.14, 2*3.14, 3*3.14}, {4*3.14, 5*3.14, 6*3.14}, {7*3.14, 8*3.14, 9*3.14}};

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.scalMult(factor);
        assertArrayEquals(exp.entries, act.entries);

        // ------------- Sub-case 2 ---------------
        aEntries = new double[][]{{-4.2345, 4.44}, {9.734, 87}, {224.5625, -90.3}, {-7.282, 0.11213}};
        factor = -3.13;
        expEntries = new double[][]{{-4.2345*-3.13, 4.44*-3.13}, {9.734*-3.13, 87*-3.13},
                {224.5625*-3.13, -90.3*-3.13}, {-7.282*-3.13, 0.11213*-3.13}};

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.scalMult(factor);
        assertArrayEquals(exp.entries, act.entries);
    }


    @Test
    void scalMultComplexTest() {
        // ------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        CFactor = new CNumber(-0.234, 12.57);
        expComplexEntries = new CNumber[][]
                {{new CNumber("-0.234+12.57i"), new CNumber("-0.468+25.14i"), new CNumber("-0.7020000000000001 + 37.71i")},
                {new CNumber("-0.936+50.28i"), new CNumber("-1.1700000000000002 + 62.85i"), new CNumber("-1.4040000000000001 + 75.42i")},
                {new CNumber("-1.6380000000000001 + 87.99000000000001i"), new CNumber("-1.872+100.56i"), new CNumber("-2.1060000000000003 + 113.13i")}};

        A = new Matrix(aEntries);
        expComplex = new CMatrix(expComplexEntries);

        actComplex = A.scalMult(CFactor);
        assertArrayEquals(expComplex.entries, actComplex.entries);

        // ------------- Sub-case 2 ---------------
        aEntries = new double[][]{{-4.2345, 4.44}, {9.734, 87}, {224.5625, -90.3}, {-7.282, 0.11213}};
        CFactor = new CNumber(4.567, -12.57);
        expComplexEntries = new CNumber[][]
                {{new CNumber("-19.3389615+53.227664999999995i"), new CNumber("20.277480000000004-55.81080000000001i")},
                {new CNumber("44.455178000000004 - 122.35638i"), new CNumber("397.329-1093.59i")},
                {new CNumber("1025.5769375-2822.750625i"), new CNumber("-412.4001+1135.071i")},
                {new CNumber("-33.256894+91.53474i"), new CNumber("0.51209771-1.4094741i")}};

        A = new Matrix(aEntries);
        expComplex = new CMatrix(expComplexEntries);

        actComplex = A.scalMult(CFactor);
        assertArrayEquals(expComplex.entries, actComplex.entries);
    }


    @Test
    void scalDivTest() {
        // ------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        factor = 3.14;
        expEntries = new double[][]{{1/3.14, 2/3.14, 3/3.14}, {4/3.14, 5/3.14, 6/3.14}, {7/3.14, 8/3.14, 9/3.14}};

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.scalDiv(factor);
        assertArrayEquals(exp.entries, act.entries);

        // ------------- Sub-case 2 ---------------
        aEntries = new double[][]{{-4.2345, 4.44}, {9.734, 87}, {224.5625, -90.3}, {-7.282, 0.11213}};
        factor = -3.13;
        expEntries = new double[][]{{-4.2345/-3.13, 4.44/-3.13}, {9.734/-3.13, 87/-3.13},
                {224.5625/-3.13, -90.3/-3.13}, {-7.282/-3.13, 0.11213/-3.13}};

        A = new Matrix(aEntries);
        exp = new Matrix(expEntries);

        act = A.scalDiv(factor);
        assertArrayEquals(exp.entries, act.entries);
    }


    @Test
    void scalDivComplexTest() {
        // ------------- Sub-case 1 ---------------
        aEntries = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        CFactor = new CNumber(-0.234, 12.57);
        expComplexEntries = new CNumber[][]
                {{new CNumber("-0.0014804536838926182 - 0.07952693507064193i"), new CNumber("-0.0029609073677852363 - 0.15905387014128386i"), new CNumber("-0.004441361051677855 - 0.23858080521192582i")},
                {new CNumber("-0.005921814735570473 - 0.3181077402825677i"), new CNumber("-0.007402268419463092 - 0.3976346753532097i"), new CNumber("-0.00888272210335571 - 0.47716161042385163i")},
                {new CNumber("-0.010363175787248328 - 0.5566885454944934i"), new CNumber("-0.011843629471140945 - 0.6362154805651354i"), new CNumber("-0.013324083155033565 - 0.7157424156357773i")}};

        A = new Matrix(aEntries);
        expComplex = new CMatrix(expComplexEntries);

        actComplex = A.scalDiv(CFactor);
        assertArrayEquals(expComplex.entries, actComplex.entries);

        // ------------- Sub-case 2 ---------------
        aEntries = new double[][]{{-4.2345, 4.44}, {9.734, 87}, {224.5625, -90.3}, {-7.282, 0.11213}};
        CFactor = new CNumber(4.567, -12.57);
        expComplexEntries = new CNumber[][]
                {{new CNumber("-0.10812201272789664 - 0.2975900372212964i"), new CNumber("0.11336916672850658 + 0.3120320616985609i")},
                        {new CNumber("0.24854402453497362 + 0.6840811010301332i"), new CNumber("2.2214228615720883 + 6.114141749498828i")},
                        {new CNumber("5.733888176457265 + 15.781689156572765i"), new CNumber("-2.3056837287351675 - 6.346057471031542i")},
                        {new CNumber("-0.1859356468731948 - 0.5117606921821893i"), new CNumber("0.002863082131816991 + 0.007880215107716132i")}};

        A = new Matrix(aEntries);
        expComplex = new CMatrix(expComplexEntries);

        actComplex = A.scalDiv(CFactor);
        assertArrayEquals(expComplex.entries, actComplex.entries);
    }
}
