"""
Security Audit Script for e-KYC Platform.
Checks for PII leaks, security vulnerabilities, and compliance issues.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class SecurityAuditor:
    """
    Security auditor for detecting PII leaks and security issues.
    """
    
    # Patterns that might indicate PII in code or logs
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    # Security anti-patterns
    SECURITY_ISSUES = {
        'hardcoded_secret': r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
        'sql_injection': r'(?:execute|query)\s*\(.*%s.*\)|f["\'].*(?:SELECT|INSERT|UPDATE|DELETE)',
        'eval_usage': r'\beval\s*\(',
        'debug_mode': r'debug\s*=\s*True',
        'print_sensitive': r'print\(.*(?:password|secret|token|key).*\)',
    }
    
    # File extensions to scan
    SCAN_EXTENSIONS = {'.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.env', '.yaml', '.yml'}
    
    # Directories to skip
    SKIP_DIRS = {'node_modules', '__pycache__', '.git', '.next', 'dist', 'build', 'venv', '.venv'}
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[Dict] = []
        self.scanned_files = 0
        self.issues_found = 0
    
    def scan_file(self, filepath: Path) -> List[Dict]:
        """Scan a single file for security issues."""
        findings = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Check for PII patterns
            for pattern_name, pattern in self.PII_PATTERNS.items():
                for i, line in enumerate(lines, 1):
                    # Skip comments and test files
                    if filepath.name.startswith('test_') or '/tests/' in str(filepath):
                        continue
                    if line.strip().startswith('#') or line.strip().startswith('//'):
                        continue
                        
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Filter false positives
                        if self._is_false_positive(pattern_name, match, line):
                            continue
                            
                        findings.append({
                            'type': 'pii_leak',
                            'category': pattern_name,
                            'file': str(filepath.relative_to(self.project_root)),
                            'line': i,
                            'snippet': line.strip()[:100],
                            'severity': 'high'
                        })
            
            # Check for security anti-patterns
            for issue_name, pattern in self.SECURITY_ISSUES.items():
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            'type': 'security_issue',
                            'category': issue_name,
                            'file': str(filepath.relative_to(self.project_root)),
                            'line': i,
                            'snippet': line.strip()[:100],
                            'severity': 'medium' if 'debug' in issue_name else 'high'
                        })
                        
        except Exception as e:
            findings.append({
                'type': 'scan_error',
                'file': str(filepath),
                'error': str(e),
                'severity': 'info'
            })
        
        return findings
    
    def _is_false_positive(self, pattern_name: str, match: str, line: str) -> bool:
        """Filter out false positives."""
        # Example email patterns in documentation
        if pattern_name == 'email':
            if 'example.com' in match or 'test' in match.lower():
                return True
        
        # Localhost IP
        if pattern_name == 'ip_address':
            if match.startswith('127.') or match.startswith('0.'):
                return True
            if match == '0.0.0.0':
                return True
                
        # Version numbers that look like IPs
        if pattern_name == 'ip_address':
            if 'version' in line.lower():
                return True
                
        return False
    
    def run_audit(self) -> Dict:
        """Run full security audit on the project."""
        print(f"Starting security audit of: {self.project_root}")
        print("-" * 50)
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            
            for filename in files:
                filepath = Path(root) / filename
                
                # Check file extension
                if filepath.suffix not in self.SCAN_EXTENSIONS:
                    continue
                
                self.scanned_files += 1
                file_findings = self.scan_file(filepath)
                
                if file_findings:
                    self.issues_found += len(file_findings)
                    self.findings.extend(file_findings)
        
        # Generate report
        report = self._generate_report()
        return report
    
    def _generate_report(self) -> Dict:
        """Generate audit report."""
        # Categorize findings
        by_severity = {'high': [], 'medium': [], 'low': [], 'info': []}
        by_type = {}
        
        for finding in self.findings:
            severity = finding.get('severity', 'info')
            by_severity[severity].append(finding)
            
            ftype = finding.get('type', 'unknown')
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append(finding)
        
        report = {
            'audit_date': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'files_scanned': self.scanned_files,
            'total_issues': self.issues_found,
            'summary': {
                'high_severity': len(by_severity['high']),
                'medium_severity': len(by_severity['medium']),
                'low_severity': len(by_severity['low']),
            },
            'findings_by_type': {k: len(v) for k, v in by_type.items()},
            'findings': self.findings,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Always recommend these
        recommendations.append("✓ Ensure all API keys and secrets are stored in environment variables")
        recommendations.append("✓ Use parameterized queries for all database operations")
        recommendations.append("✓ Enable HTTPS in production")
        recommendations.append("✓ Implement rate limiting on all endpoints")
        recommendations.append("✓ Set up proper CORS configuration")
        recommendations.append("✓ Enable audit logging for all KYC operations")
        
        # Based on findings
        has_pii = any(f.get('type') == 'pii_leak' for f in self.findings)
        has_security = any(f.get('type') == 'security_issue' for f in self.findings)
        
        if has_pii:
            recommendations.append("⚠ CRITICAL: Remove or mask all PII from code and logs")
            
        if has_security:
            recommendations.append("⚠ Review and fix security anti-patterns identified")
        
        if self.issues_found == 0:
            recommendations.insert(0, "✓ No critical security issues detected in code scan")
        
        return recommendations
    
    def print_report(self, report: Dict):
        """Print human-readable report."""
        print("\n" + "=" * 60)
        print("SECURITY AUDIT REPORT")
        print("=" * 60)
        print(f"Date: {report['audit_date']}")
        print(f"Files Scanned: {report['files_scanned']}")
        print(f"Total Issues: {report['total_issues']}")
        print()
        
        print("SEVERITY SUMMARY:")
        print(f"  🔴 High:   {report['summary']['high_severity']}")
        print(f"  🟡 Medium: {report['summary']['medium_severity']}")
        print(f"  🟢 Low:    {report['summary']['low_severity']}")
        print()
        
        if report['findings']:
            print("FINDINGS:")
            for finding in report['findings'][:10]:  # Show first 10
                severity_icon = '🔴' if finding.get('severity') == 'high' else '🟡'
                print(f"  {severity_icon} [{finding.get('category', 'unknown')}] {finding.get('file', '')}:{finding.get('line', '')}")
                print(f"     {finding.get('snippet', '')[:80]}...")
            
            if len(report['findings']) > 10:
                print(f"  ... and {len(report['findings']) - 10} more findings")
        
        print()
        print("RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print()
        print("=" * 60)


def main():
    """Run security audit."""
    import sys
    
    # Default to parent directory (ekyc-platform)
    project_root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent)
    
    auditor = SecurityAuditor(project_root)
    report = auditor.run_audit()
    auditor.print_report(report)
    
    # Save report
    report_path = Path(project_root) / 'security_audit_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")
    
    # Return exit code based on severity
    if report['summary']['high_severity'] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
