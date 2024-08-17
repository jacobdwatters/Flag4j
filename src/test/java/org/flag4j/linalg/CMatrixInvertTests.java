package org.flag4j.linalg;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CMatrixInvertTests {

    static CMatrixOld A;
    static CMatrixOld exp;
    static CNumber[][] entries;
    static CNumber[][] expEntries;


    @Test
    void invTriUTests() {
        // --------------------- Sub-case 1 ---------------------
        entries = new CNumber[][]{
                {new CNumber("0.72964+0.77161i"), new CNumber("0.04017+0.35861i")},
                {new CNumber("0.0"), new CNumber("0.02885+0.61375i")}};
        A = new CMatrixOld(entries);
        expEntries = new CNumber[][]{
                {new CNumber(0.6469836227593948, -0.6841991025127141), new CNumber(-0.35324900268232806, 0.4255132196349395)},
                {new CNumber(0.0, 0.0), new CNumber(0.07641951197016296, -1.6257357182560666)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, Invert.invTriU(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new CNumber[][]{
                {new CNumber("0.88984+0.74576i"), new CNumber("0.35899+0.67567i"), new CNumber("0.67057+0.03486i"), new CNumber("0.27239+0.14667i")},
                {new CNumber("0.0"), new CNumber("0.29455+0.74263i"), new CNumber("0.99331+0.19522i"), new CNumber("0.89592+0.76741i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.70384+0.74177i"), new CNumber("0.59081+0.73518i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.97421+0.61448i")}};
        A = new CMatrixOld(entries);
        expEntries = new CNumber[][]{
                {new CNumber(0.6601318170773559, -0.5532454192929167), new CNumber(-0.569755474276714, 0.5964889359681658), new CNumber(0.0538261672936495, -0.24610680231262014), new CNumber(0.39210142039761287, -0.18312414720644973)},
                {new CNumber(0.0, 0.0), new CNumber(0.46148975736667824, -1.1635244899447168), new CNumber(0.29451765428159893, 1.2036600935687118), new CNumber(-0.5481358478041402, 0.10001294703720921)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.6731359287847971, -0.709411283664894), new CNumber(-0.7101078401778858, 0.3701442953886289)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.7343268609204316, -0.4631744382611417)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, Invert.invTriU(A));
    }


    @Test
    void invTriLTests() {
        // --------------------- Sub-case 1 ---------------------
        entries = new CNumber[][]{
                {new CNumber("0.72964+0.77161i"), new CNumber("0.04017+0.35861i")},
                {new CNumber("0.0"), new CNumber("0.02885+0.61375i")}};
        A = new CMatrixOld(entries).T();
        expEntries = new CNumber[][]{
                {new CNumber(0.6469836227593948, -0.6841991025127141), new CNumber(-0.35324900268232806, 0.4255132196349395)},
                {new CNumber(0.0, 0.0), new CNumber(0.07641951197016296, -1.6257357182560666)}};
        exp = new CMatrixOld(expEntries).T();

        assertEquals(exp, Invert.invTriL(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new CNumber[][]{
                {new CNumber("0.88984+0.74576i"), new CNumber("0.35899+0.67567i"), new CNumber("0.67057+0.03486i"), new CNumber("0.27239+0.14667i")},
                {new CNumber("0.0"), new CNumber("0.29455+0.74263i"), new CNumber("0.99331+0.19522i"), new CNumber("0.89592+0.76741i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.70384+0.74177i"), new CNumber("0.59081+0.73518i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.97421+0.61448i")}};
        A = new CMatrixOld(entries).T();
        expEntries = new CNumber[][]{
                {new CNumber(0.6601318170773559, -0.5532454192929167), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(-0.5697554742767139, 0.5964889359681657), new CNumber(0.46148975736667824, -1.1635244899447168), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.05382616729364944, -0.2461068023126199), new CNumber(0.29451765428159893, 1.203660093568712), new CNumber(0.6731359287847971, -0.709411283664894), new CNumber(0.0, 0.0)},
                {new CNumber(0.39210142039761287, -0.18312414720644984), new CNumber(-0.5481358478041398, 0.10001294703720916), new CNumber(-0.7101078401778858, 0.3701442953886289), new CNumber(0.7343268609204316, -0.4631744382611417)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, Invert.invTriL(A));
    }


    @Test
    void invDiagTests() {
        // --------------------- Sub-case 1 ---------------------
        entries = new CNumber[][]{
                {new CNumber(-14.43, 95.1), CNumber.ZERO},
                {CNumber.ZERO, new CNumber(0, 1.45)}};
        A = new CMatrixOld(entries);
        expEntries = new CNumber[][]{
                {new CNumber(-14.43, 95.1).multInv(), CNumber.ZERO},
                {CNumber.ZERO, new CNumber(0, 1.45).multInv()}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, Invert.invDiag(A));

        // --------------------- Sub-case 2 ---------------------
        entries = new CNumber[][]{
                {new CNumber(-14.43, 95.1), CNumber.ZERO, CNumber.ZERO},
                {CNumber.ZERO, new CNumber(0, 1.45), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(234.156)}};
        A = new CMatrixOld(entries);
        expEntries = new CNumber[][]{
                {new CNumber(-14.43, 95.1).multInv(), CNumber.ZERO, CNumber.ZERO},
                {CNumber.ZERO, new CNumber(0, 1.45).multInv(), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(234.156).multInv()}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, Invert.invDiag(A));
    }
}
