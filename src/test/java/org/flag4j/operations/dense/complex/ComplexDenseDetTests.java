package org.flag4j.operations.dense.complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexDenseDetTests {

    CNumber[][] entries;
    CMatrix A;  // Matrix to compute determinate of.
    CNumber act;  // Stores actual computed determinate.
    CNumber exp;  // Stores expected determinate.

    @Test
    void det1Test() {
        // -------------- Sub-case 1 --------------
        entries = new CNumber[][]{{new CNumber(9.5, -14.5)}};
        A = new CMatrix(entries);
        exp = new CNumber(9.5, -14.5);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        entries = new CNumber[][]{{new CNumber(1.4)}};
        A = new CMatrix(entries);
        exp = new CNumber(1.4);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        entries = new CNumber[][]{{new CNumber(-9)}};
        A = new CMatrix(entries);
        exp = new CNumber(-9);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        entries = new CNumber[][]{{new CNumber(0, 9765.134523400202002)}};
        A = new CMatrix(entries);
        exp = new CNumber(0, 9765.134523400202002);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        entries = new CNumber[][]{{new CNumber(), new CNumber()}, {new CNumber(), new CNumber()}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det1(A));
    }


    @Test
    void det2Test() {
        // -------------- Sub-case 1 --------------
        entries = new CNumber[][]{
                {new CNumber("-1.0"), new CNumber("5.0")},
                {new CNumber("133.5"), new CNumber("25.0")}};
        A = new CMatrix(entries);
        exp = new CNumber(-692.5);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        entries = new CNumber[][]{
                {new CNumber("-1.0+0.345i"), new CNumber("5.0+2.0i")},
                {new CNumber("133.5+0.00014i"), new CNumber("-0.0-25.0i")}};
        A = new CMatrix(entries);
        exp = new CNumber("-658.87472 - 242.0007i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        entries = new CNumber[][]{
                {new CNumber("0.0+8.0i"), new CNumber("-3.0+2.0i")},
                {new CNumber("-3.0+2.0i"), new CNumber("-0.0-25.0i")}};
        A = new CMatrix(entries);
        exp = new CNumber("195+12i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        entries = new CNumber[][]{
                {new CNumber("1.0"), new CNumber("6.0-2.0i")},
                {new CNumber("6.0+2.0i"), new CNumber("5.0")}};
        A = new CMatrix(entries);
        exp = new CNumber("-35");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        entries = new CNumber[][]{{new CNumber()}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det2(A));

        // -------------- Sub-case 6 --------------
        entries = new CNumber[][]{
                {new CNumber(), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber()}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det2(A));
    }


    @Test
    void det3Test() {
        // -------------- Sub-case 1 --------------
        entries = new CNumber[][]{
                {new CNumber("1"), new CNumber("2"), new CNumber("3")},
                {new CNumber("4"), new CNumber("5"), new CNumber("6")},
                {new CNumber("7"), new CNumber("8"), new CNumber("9")}};
        A = new CMatrix(entries);
        exp = new CNumber();
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        entries = new CNumber[][]{
                {new CNumber("19.45"), new CNumber("-0.0035"), new CNumber("125.6")},
                {new CNumber("1.000035"), new CNumber("5.6"), new CNumber("1556.78679")},
                {new CNumber("6.0"), new CNumber("-992.45"), new CNumber("15.7")}};
        A = new CMatrix(entries);
        exp = new CNumber("29923693.980974607");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        entries = new CNumber[][]{
                {new CNumber("19.45+0.02026i"), new CNumber("-0.0035+8.0i"), new CNumber("125.6")},
                {new CNumber("1.000035"), new CNumber("5.6+30.49856i"), new CNumber("1556.78679-985.355i")},
                {new CNumber("6.0-2.0i"), new CNumber("-992.45-2.0i"), new CNumber("15.7+0.091456i")}};
        A = new CMatrix(entries);
        exp = new CNumber("30046261.835600328 - 18882193.78222524i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 4 --------------
        entries = new CNumber[][]{{new CNumber(), new CNumber()}, {new CNumber(), new CNumber()}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det3(A));

        // -------------- Sub-case 5 --------------
        entries = new CNumber[][]{
                {new CNumber(), new CNumber(), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber(), new CNumber()}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det3(A));
    }

    @Test
    void detLUTest() {
        // -------------- Sub-case 1 --------------
        entries = new CNumber[][]{
                {new CNumber("1.0"), new CNumber("-53.0"), new CNumber("611.5"), new CNumber("234.6")},
                {new CNumber("-0.00351"), new CNumber("0.0"), new CNumber("24.56"), new CNumber("14.6")},
                {new CNumber("998.3"), new CNumber("2.5"), new CNumber("1.0"), new CNumber("0.00000025")},
                {new CNumber("-92.5"), new CNumber("14.5"), new CNumber("7.1"), new CNumber("34.0")}};
        A = new CMatrix(entries);
        exp = new CNumber("7935026.786550307");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 2 --------------
        entries = new CNumber[][]{
                {new CNumber("0.41059+0.9284i"), new CNumber("0.13828+0.91072i"), new CNumber("0.70458+0.9232i"), new CNumber("0.6675+0.94988i")},
                {new CNumber("0.46771+0.6101i"), new CNumber("0.82633+0.28485i"), new CNumber("0.22087+0.69659i"), new CNumber("0.97234+0.36372i")},
                {new CNumber("0.97155+0.84824i"), new CNumber("0.13442+0.81678i"), new CNumber("0.65467+0.31053i"), new CNumber("0.62112+0.6425i")},
                {new CNumber("0.68325+0.50348i"), new CNumber("0.50074+0.96227i"), new CNumber("0.84584+0.45291i"), new CNumber("0.57167+0.06987i")}};
        A = new CMatrix(entries);
        exp = new CNumber("-0.33775706366650027+0.46301856289254556i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 3 --------------
        entries = new CNumber[][]{
                {new CNumber("0.00399+0.99393i"), new CNumber("0.5602+0.87092i"), new CNumber("0.94279+0.72353i"), new CNumber("0.26672+0.89313i"), new CNumber("0.93043+0.36442i"), new CNumber("0.56652+0.75951i"), new CNumber("0.38806+0.72435i"), new CNumber("0.34656+0.05705i"), new CNumber("0.98495+0.98219i"), new CNumber("0.19058+0.99072i"), new CNumber("0.71766+0.87942i"), new CNumber("0.67899+0.64893i"), new CNumber("0.57014+0.89789i"), new CNumber("0.35228+0.72217i"), new CNumber("0.91898+0.55195i")},
                {new CNumber("0.83284+0.56167i"), new CNumber("0.72028+0.14505i"), new CNumber("0.84075+0.29224i"), new CNumber("0.93349+0.37263i"), new CNumber("0.97302+0.3842i"), new CNumber("0.881+0.79757i"), new CNumber("0.83075+0.93624i"), new CNumber("0.92814+0.47567i"), new CNumber("0.26395+0.93256i"), new CNumber("0.01236+0.38783i"), new CNumber("0.1645+0.91338i"), new CNumber("0.5939+0.55956i"), new CNumber("0.41956+0.05676i"), new CNumber("0.16001+0.19826i"), new CNumber("0.65655+0.63476i")},
                {new CNumber("0.64889+0.68114i"), new CNumber("0.20075+0.60461i"), new CNumber("0.03752+0.54424i"), new CNumber("0.16755+0.49303i"), new CNumber("0.08295+0.81619i"), new CNumber("0.54819+0.34945i"), new CNumber("0.61779+0.65259i"), new CNumber("0.32885+0.34203i"), new CNumber("0.88752+0.02295i"), new CNumber("0.74886+0.94301i"), new CNumber("0.46661+0.46275i"), new CNumber("0.0622+0.06144i"), new CNumber("0.12673+0.08713i"), new CNumber("0.36444+0.22611i"), new CNumber("0.8904+0.09787i")},
                {new CNumber("0.08212+0.21184i"), new CNumber("0.77892+0.8349i"), new CNumber("0.06279+0.86456i"), new CNumber("0.83241+0.71363i"), new CNumber("0.45711+0.5999i"), new CNumber("0.60289+0.7781i"), new CNumber("0.22582+0.91327i"), new CNumber("0.9733+0.61772i"), new CNumber("0.61748+0.438i"), new CNumber("0.02977+0.65873i"), new CNumber("0.64303+0.19697i"), new CNumber("0.45741+0.53814i"), new CNumber("0.50226+0.34232i"), new CNumber("0.9652+0.07049i"), new CNumber("0.84176+0.60646i")},
                {new CNumber("0.9111+0.32226i"), new CNumber("0.91951+0.93551i"), new CNumber("0.239+0.48126i"), new CNumber("0.16065+0.63575i"), new CNumber("0.71097+0.72539i"), new CNumber("0.10342+0.22849i"), new CNumber("0.96078+0.55933i"), new CNumber("0.98355+0.44822i"), new CNumber("0.96949+0.91869i"), new CNumber("0.58076+0.45637i"), new CNumber("0.90556+0.04729i"), new CNumber("0.26906+0.77125i"), new CNumber("0.168+0.82065i"), new CNumber("0.06128+0.08468i"), new CNumber("0.35874+0.97037i")},
                {new CNumber("0.77575+0.3898i"), new CNumber("0.7468+0.58737i"), new CNumber("0.75703+0.43206i"), new CNumber("0.00236+0.2227i"), new CNumber("0.69134+0.50974i"), new CNumber("0.23623+0.08836i"), new CNumber("0.7311+0.49747i"), new CNumber("0.44573+0.63388i"), new CNumber("0.21488+0.76934i"), new CNumber("0.10153+0.46149i"), new CNumber("0.68971+0.63242i"), new CNumber("0.22933+0.37297i"), new CNumber("0.70293+0.71534i"), new CNumber("0.01343+0.76882i"), new CNumber("0.84475+0.50923i")},
                {new CNumber("0.82659+0.68179i"), new CNumber("0.63321+0.89338i"), new CNumber("0.37323+0.10609i"), new CNumber("0.77274+0.88217i"), new CNumber("0.57556+0.79726i"), new CNumber("0.73872+0.10875i"), new CNumber("0.82994+0.44293i"), new CNumber("0.8565+0.50379i"), new CNumber("0.85324+0.31997i"), new CNumber("0.35543+0.44813i"), new CNumber("0.16575+0.45721i"), new CNumber("0.30071+0.71059i"), new CNumber("0.91801+0.05638i"), new CNumber("0.98455+0.75933i"), new CNumber("0.03145+0.84904i")},
                {new CNumber("0.80872+0.41368i"), new CNumber("0.43987+0.81422i"), new CNumber("0.25294+0.16763i"), new CNumber("0.2439+0.09514i"), new CNumber("0.25768+0.7513i"), new CNumber("0.31547+0.86362i"), new CNumber("0.60704+0.8838i"), new CNumber("0.36612+0.06238i"), new CNumber("0.0969+0.59707i"), new CNumber("0.96099+0.87713i"), new CNumber("0.28446+0.45232i"), new CNumber("0.95354+0.87055i"), new CNumber("0.6418+0.68782i"), new CNumber("0.73455+0.04973i"), new CNumber("0.71273+0.48545i")},
                {new CNumber("0.8469+0.80951i"), new CNumber("0.71347+0.76274i"), new CNumber("0.35338+0.97138i"), new CNumber("0.50707+0.28844i"), new CNumber("0.23745+0.73877i"), new CNumber("0.82702+0.8458i"), new CNumber("0.08864+0.39934i"), new CNumber("0.02697+0.55562i"), new CNumber("0.20154+0.59157i"), new CNumber("0.6372+0.50199i"), new CNumber("0.98408+0.68102i"), new CNumber("0.32651+0.76286i"), new CNumber("0.19108+0.25295i"), new CNumber("0.25825+0.35311i"), new CNumber("0.75368+0.72706i")},
                {new CNumber("0.34448+0.04438i"), new CNumber("0.30705+0.53208i"), new CNumber("0.36304+0.99933i"), new CNumber("0.68529+0.11284i"), new CNumber("0.38368+0.8677i"), new CNumber("0.19719+0.13337i"), new CNumber("0.43275+0.56616i"), new CNumber("0.68267+0.99983i"), new CNumber("0.89103+0.16879i"), new CNumber("0.61772+0.67517i"), new CNumber("0.1031+0.83389i"), new CNumber("0.2915+0.45911i"), new CNumber("0.02961+0.30488i"), new CNumber("0.48733+0.58809i"), new CNumber("0.49888+0.6453i")},
                {new CNumber("0.75839+0.61182i"), new CNumber("0.76859+0.50697i"), new CNumber("0.28901+0.4631i"), new CNumber("0.63993+0.58199i"), new CNumber("0.76412+0.1298i"), new CNumber("0.31395+0.22497i"), new CNumber("0.20133+0.35658i"), new CNumber("0.81514+0.55195i"), new CNumber("0.66929+0.33394i"), new CNumber("0.12604+0.23074i"), new CNumber("0.47876+0.13968i"), new CNumber("0.88261+0.02473i"), new CNumber("0.15945+0.15319i"), new CNumber("0.09627+0.83235i"), new CNumber("0.11175+0.68569i")},
                {new CNumber("0.28752+0.19156i"), new CNumber("0.38266+0.30584i"), new CNumber("0.48181+0.10724i"), new CNumber("0.3457+0.31926i"), new CNumber("0.53183+0.37206i"), new CNumber("0.27003+0.4516i"), new CNumber("0.94186+0.43975i"), new CNumber("0.60834+0.37273i"), new CNumber("0.48761+0.48447i"), new CNumber("0.12911+0.98924i"), new CNumber("0.10905+0.72429i"), new CNumber("0.8166+0.76886i"), new CNumber("0.92743+0.06145i"), new CNumber("0.45529+0.97434i"), new CNumber("0.79746+0.75906i")},
                {new CNumber("0.38163+0.21341i"), new CNumber("0.92648+0.56462i"), new CNumber("0.0785+0.47142i"), new CNumber("0.46961+0.6336i"), new CNumber("0.73251+0.90862i"), new CNumber("0.87166+0.25754i"), new CNumber("0.6156+0.18451i"), new CNumber("0.53142+0.52466i"), new CNumber("0.90182+0.27299i"), new CNumber("0.79387+0.35015i"), new CNumber("0.12427+0.29163i"), new CNumber("0.31682+0.14227i"), new CNumber("0.18861+0.52372i"), new CNumber("0.47167+0.75696i"), new CNumber("0.86124+0.8967i")},
                {new CNumber("0.63451+0.55208i"), new CNumber("0.30552+0.90471i"), new CNumber("0.18693+0.52159i"), new CNumber("0.95717+0.41237i"), new CNumber("0.93016+0.78026i"), new CNumber("0.74042+0.8602i"), new CNumber("0.42962+0.4916i"), new CNumber("0.67385+0.18248i"), new CNumber("0.53536+0.84218i"), new CNumber("0.45487+0.74388i"), new CNumber("0.87338+0.13472i"), new CNumber("0.5436+0.17422i"), new CNumber("0.19605+0.12928i"), new CNumber("0.28215+0.03012i"), new CNumber("0.88307+0.24797i")},
                {new CNumber("0.82246+0.03127i"), new CNumber("0.59935+0.32426i"), new CNumber("0.94404+0.28634i"), new CNumber("0.93376+0.44786i"), new CNumber("0.40843+0.65209i"), new CNumber("0.08543+0.05725i"), new CNumber("0.35585+0.95449i"), new CNumber("0.43067+0.33572i"), new CNumber("0.39394+0.05754i"), new CNumber("0.44095+0.56715i"), new CNumber("0.02149+0.01673i"), new CNumber("0.2843+0.81159i"), new CNumber("0.96+0.85875i"), new CNumber("0.29013+0.25358i"), new CNumber("0.03828+0.06116i")}};
        A = new CMatrix(entries);
        exp = new CNumber("3.59504116227108-5.327292514154561i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- Sub-case 5 --------------
        entries = new CNumber[][]{
                {new CNumber("1.0"), new CNumber("-53.0"), new CNumber("611.5"), new CNumber("234.6")},
                {new CNumber("-0.00351"), new CNumber("0.0"), new CNumber("24.56"), new CNumber("14.6")},
                {new CNumber("998.3"), new CNumber("2.5"), new CNumber("1.0"), new CNumber("0.00000025")}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det(A));

        // -------------- Sub-case 6 --------------
        entries = new CNumber[][]{
                {new CNumber("1.0"), new CNumber("-53.0"), new CNumber("611.5")},
                {new CNumber("-0.00351"), new CNumber("0.0"), new CNumber("24.56")},
                {new CNumber("998.3"), new CNumber("2.5"), new CNumber("1.0")},
                {new CNumber("-92.5"), new CNumber("14.5"), new CNumber("7.1")}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det(A));
    }
}
