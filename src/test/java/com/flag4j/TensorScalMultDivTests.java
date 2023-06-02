package com.flag4j;

import com.flag4j.CTensor;
import com.flag4j.Shape;
import com.flag4j.Tensor;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorScalMultDivTests {

    static double[] aEntries;
    static Tensor A;
    static Shape aShape, expShape;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realScalMultTest() {
        double[] expEntries;
        Tensor exp;
        double b = -1.4115;

        // ------------------------ Sub-case 1 ------------------------
        expEntries = new double[]{
                1.23*-1.4115, 2.556*-1.4115, -121.5*-1.4115, 15.61*-1.4115, 14.15*-1.4115, -99.23425*-1.4115,
                0.001345*-1.4115, 2.677*-1.4115, 8.14*-1.4115, -0.000194*-1.4115, 1*-1.4115, 234*-1.4115
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.mult(b));
    }


    @Test
    void complexScalMultTest() {
        CNumber[] expEntries;
        CTensor exp;
        CNumber b = new CNumber(0.2425, -0.00295);

        // ------------------------ Sub-case 1 ------------------------
        expEntries = new CNumber[]{
                b.mult(1.23), b.mult(2.556), b.mult(-121.5), b.mult(15.61), b.mult(14.15), b.mult(-99.23425),
                b.mult(0.001345), b.mult(2.677), b.mult(8.14), b.mult(-0.000194), b.mult(1), b.mult(234)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.mult(b));
    }


    @Test
    void realScalDivTest() {
        double[] expEntries;
        Tensor exp;
        double b = -1.4115;

        // ------------------------ Sub-case 1 ------------------------
        expEntries = new double[]{
                1.23/-1.4115, 2.556/-1.4115, -121.5/-1.4115, 15.61/-1.4115, 14.15/-1.4115, -99.23425/-1.4115,
                0.001345/-1.4115, 2.677/-1.4115, 8.14/-1.4115, -0.000194/-1.4115, 1/-1.4115, 234/-1.4115
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.div(b));
    }


    @Test
    void complexScalDivTest() {
        CNumber[] expEntries;
        CTensor exp;
        CNumber b = new CNumber(0.2425, -0.00295);
        CNumber bInv = b.multInv();

        // ------------------------ Sub-case 1 ------------------------
        expEntries = new CNumber[]{
                new CNumber("5.071414450262457 + 0.06169349537432679i"),
                new CNumber("10.53864661371613 + 0.12820209282664985i"),
                bInv.mult(-121.5),
                bInv.mult(15.61),
                new CNumber("58.341881683913634 + 0.7097259833713205i"),
                new CNumber("-409.15285317964 - 4.977323368577064i"),
                bInv.mult(0.001345),
                new CNumber("11.037541856384227 + 0.13427112773745759i"),
                new CNumber("33.562043597671874 + 0.408280530363431i"),
                bInv.mult(-0.000194),
                bInv.mult(1),
                bInv.mult(234)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.div(b));
    }
}
