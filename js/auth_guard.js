/**
 * ZuraStock Authentication Guard
 * Protects pages from unauthorized access and handles user session.
 */

(function () {
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    const isLoginPage = window.location.pathname.endsWith('login.html');
    const isRegisterPage = window.location.pathname.endsWith('register.html');

    // If no user and not on login/register, redirect to login
    if (!currentUser && !isLoginPage && !isRegisterPage) {
        console.log('AuthGuard: No user found, redirecting to login...');
        window.location.href = 'login.html';
        return;
    }

    // If user is logged in and on login/register, redirect to dashboard
    if (currentUser && (isLoginPage || isRegisterPage)) {
        window.location.href = 'index.html';
        return;
    }

    // Global Logout Function
    window.logout = function () {
        localStorage.removeItem('currentUser');
        window.location.href = 'login.html';
    };

    // Inject user name into UI if elements exist
    document.addEventListener('DOMContentLoaded', () => {
        if (currentUser) {
            const userNameEls = document.querySelectorAll('.user-name-display, #userName');
            const userEmailEls = document.querySelectorAll('.user-email-display, #userEmail');
            
            userNameEls.forEach(el => el.textContent = currentUser.username || currentUser.name);
            userEmailEls.forEach(el => el.textContent = currentUser.email);
        }
    });
})();
